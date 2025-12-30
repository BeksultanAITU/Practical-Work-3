#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)

static inline int iDivUp(int a, int b){ return (a+b-1)/b; }

static bool is_sorted_cpu(const std::vector<int>& a){
  for(size_t i=1;i<a.size();++i) if(a[i-1]>a[i]) return false;
  return true;
}

static float elapsed_ms(cudaEvent_t s, cudaEvent_t t){
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms,s,t)); return ms;
}

/* =========================
   Task 1: CUDA Merge Sort
   ========================= */

template<int CHUNK>
__global__ void bitonicSortChunks(int* data, int n){
  // Делим массив на куски CHUNK: один CUDA-блок на один кусок.
  __shared__ int s[CHUNK];

  int start = blockIdx.x * CHUNK;
  int tid = threadIdx.x;

  // Загружаем в shared: по 2 элемента на поток, хвост заполняем INT32_MAX.
  int i0 = start + 2*tid;
  int i1 = start + 2*tid + 1;
  s[2*tid]     = (i0<n) ? data[i0] : INT32_MAX;
  s[2*tid + 1] = (i1<n) ? data[i1] : INT32_MAX;
  __syncthreads();

  // Локально сортируем кусок битоникой внутри shared памяти.
  for(int k=2;k<=CHUNK;k<<=1){
    for(int j=k>>1;j>0;j>>=1){
      int base = 2*tid;
      for(int t=0;t<2;++t){
        int i = base+t;
        int ixj = i ^ j;
        if(ixj>i){
          bool up = ((i & k)==0);
          int a=s[i], b=s[ixj];
          if((a>b)==up){ s[i]=b; s[ixj]=a; }
        }
      }
      __syncthreads();
    }
  }

  // Записываем обратно отсортированный кусок.
  if(i0<n) data[i0]=s[2*tid];
  if(i1<n) data[i1]=s[2*tid+1];
}

__device__ __forceinline__ int mymin(int a,int b){ return a<b?a:b; }

__global__ void mergePairsKernel(const int* in, int* out, int n, int width){
  // После локальной сортировки делаем проходы слияния: CHUNK, 2*CHUNK, 4*CHUNK...
  int start = (int)blockIdx.x * 2 * width;
  if(start>=n) return;

  int mid = mymin(start + width, n);
  int end = mymin(start + 2*width, n);

  // Один блок делает одно слияние двух отсортированных отрезков.
  if(threadIdx.x==0){
    int i=start, j=mid, k=start;
    while(i<mid && j<end){
      int a=in[i], b=in[j];
      if(a<=b) out[k++]=a, ++i;
      else     out[k++]=b, ++j;
    }
    while(i<mid) out[k++]=in[i++];
    while(j<end) out[k++]=in[j++];
  }
}

void gpu_merge_sort(std::vector<int>& a){
  int n=(int)a.size(); if(!n) return;
  constexpr int CHUNK=1024;

  int *dA=nullptr,*dB=nullptr;
  CUDA_CHECK(cudaMalloc(&dA,n*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dB,n*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dA,a.data(),n*sizeof(int),cudaMemcpyHostToDevice));

  // 1) Сортируем куски параллельно: каждый блок потоков свой кусок.
  int numChunks=iDivUp(n,CHUNK);
  bitonicSortChunks<CHUNK><<<numChunks, CHUNK/2>>>(dA,n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 2) Сливаем куски попарно, удваивая ширину на каждом шаге.
  int width=CHUNK;
  bool flip=false;
  while(width<n){
    const int* in = flip ? dB : dA;
    int* out      = flip ? dA : dB;

    int pairs=iDivUp(n,2*width);
    mergePairsKernel<<<pairs,256>>>(in,out,n,width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    flip=!flip;
    width*=2;
  }

  const int* dRes = flip ? dB : dA;
  CUDA_CHECK(cudaMemcpy(a.data(),dRes,n*sizeof(int),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
}

void cpu_merge_sort(std::vector<int>& a){
  // Последовательная версия на CPU.
  std::stable_sort(a.begin(),a.end());
}

/* =========================
   Task 2: CUDA Quick Sort
   ========================= */

__global__ void countLessKernel(const int* in, int l, int r, int pivot, int* cnt){
  // Параллельно считаем сколько элементов < pivot.
  int idx = l + (int)(blockIdx.x*blockDim.x + threadIdx.x);
  if(idx<r && in[idx]<pivot) atomicAdd(cnt,1);
}

__global__ void scatterKernel(const int* in, int* out, int l, int r, int pivot,
                             int* leftPos, int* rightPos, int baseRight){
  // Параллельно раскладываем: < pivot влево, остальное вправо.
  int idx = l + (int)(blockIdx.x*blockDim.x + threadIdx.x);
  if(idx>=r) return;
  int v=in[idx];
  if(v<pivot){
    int p=atomicAdd(leftPos,1);
    out[l+p]=v;
  }else{
    int p=atomicAdd(rightPos,1);
    out[baseRight+p]=v;
  }
}

template<int CHUNK>
__global__ void bitonicSortSegment(int* data, int n, int l, int r){
  // Маленькие сегменты досортировываем целиком в одном блоке.
  __shared__ int s[CHUNK];
  int tid=threadIdx.x;
  int len=r-l;

  s[tid] = (tid<len && l+tid<n) ? data[l+tid] : INT32_MAX;
  __syncthreads();

  for(int k=2;k<=CHUNK;k<<=1){
    for(int j=k>>1;j>0;j>>=1){
      int ixj=tid^j;
      if(ixj>tid){
        bool up=((tid&k)==0);
        int a=s[tid], b=s[ixj];
        if((a>b)==up){ s[tid]=b; s[ixj]=a; }
      }
      __syncthreads();
    }
  }
  if(tid<len && l+tid<n) data[l+tid]=s[tid];
}

void gpu_quick_sort(std::vector<int>& a){
  int n=(int)a.size(); if(!n) return;
  constexpr int SMALL=1024;

  int *dA=nullptr,*dTmp=nullptr,*dCnt=nullptr,*dL=nullptr,*dR=nullptr;
  CUDA_CHECK(cudaMalloc(&dA,n*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dTmp,n*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dCnt,sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dL,sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dR,sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dA,a.data(),n*sizeof(int),cudaMemcpyHostToDevice));

  // Рекурсию держим стеком на CPU, а partition делаем параллельно на GPU.
  std::vector<std::pair<int,int>> st;
  st.push_back({0,n});

  while(!st.empty()){
    auto [l,r]=st.back(); st.pop_back();
    int len=r-l;
    if(len<=1) continue;

    // Малые куски сортируем быстро внутри одного блока.
    if(len<=SMALL){
      bitonicSortSegment<SMALL><<<1,SMALL>>>(dA,n,l,r);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      continue;
    }

    // Pivot берём из середины сегмента.
    int mid=l+len/2;
    int pivot=0;
    CUDA_CHECK(cudaMemcpy(&pivot,dA+mid,sizeof(int),cudaMemcpyDeviceToHost));

    // 1) Считаем количество элементов < pivot.
    CUDA_CHECK(cudaMemset(dCnt,0,sizeof(int)));
    int threads=256, blocks=iDivUp(len,threads);
    countLessKernel<<<blocks,threads>>>(dA,l,r,pivot,dCnt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int cntLess=0;
    CUDA_CHECK(cudaMemcpy(&cntLess,dCnt,sizeof(int),cudaMemcpyDeviceToHost));

    // 2) Раскладываем элементы по двум зонам параллельно.
    CUDA_CHECK(cudaMemset(dL,0,sizeof(int)));
    CUDA_CHECK(cudaMemset(dR,0,sizeof(int)));
    int baseRight=l+cntLess;
    scatterKernel<<<blocks,threads>>>(dA,dTmp,l,r,pivot,dL,dR,baseRight);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3) Возвращаем partition обратно в основной массив.
    CUDA_CHECK(cudaMemcpy(dA+l,dTmp+l,len*sizeof(int),cudaMemcpyDeviceToDevice));

    st.push_back({l,l+cntLess});
    st.push_back({l+cntLess,r});
  }

  CUDA_CHECK(cudaMemcpy(a.data(),dA,n*sizeof(int),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dTmp));
  CUDA_CHECK(cudaFree(dCnt));
  CUDA_CHECK(cudaFree(dL));
  CUDA_CHECK(cudaFree(dR));
}

void cpu_quick_sort(std::vector<int>& a){
  // Последовательная версия на CPU.
  std::sort(a.begin(),a.end());
}

/* =========================
   Task 3: CUDA Heap Sort
   ========================= */

__device__ void siftDown(int* a, int n, int i){
  // Просеивание вниз: поддерживаем свойство кучи.
  while(true){
    int l=2*i+1, r=l+1, best=i;
    if(l<n && a[l]>a[best]) best=l;
    if(r<n && a[r]>a[best]) best=r;
    if(best==i) break;
    int t=a[i]; a[i]=a[best]; a[best]=t;
    i=best;
  }
}

__global__ void heapifyLevelKernel(int* a, int n, int firstIdx, int count){
  // На одном уровне дерева узлы можно просеивать параллельно.
  int t=(int)(blockIdx.x*blockDim.x + threadIdx.x);
  if(t>=count) return;
  siftDown(a,n,firstIdx+t);
}

__global__ void heapExtractKernel(int* a, int n){
  // Извлечение максимума почти последовательное, делаем одним потоком.
  if(blockIdx.x||threadIdx.x) return;
  for(int end=n-1; end>0; --end){
    int t=a[0]; a[0]=a[end]; a[end]=t;
    siftDown(a,end,0);
  }
}

void gpu_heap_sort(std::vector<int>& a){
  int n=(int)a.size(); if(!n) return;
  int* dA=nullptr;
  CUDA_CHECK(cudaMalloc(&dA,n*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dA,a.data(),n*sizeof(int),cudaMemcpyHostToDevice));

  // 1) Строим кучу снизу вверх по уровням, на каждом уровне параллельно.
  int lastInternal=n/2-1;
  if(lastInternal>=0){
    int d=0;
    while(((1<<(d+1))-1)<=lastInternal) ++d;

    for(int level=d; level>=0; --level){
      int start=(1<<level)-1;
      int end=(1<<(level+1))-2;
      if(start>lastInternal) continue;
      if(end>lastInternal) end=lastInternal;

      int count=end-start+1;
      int threads=256, blocks=iDivUp(count,threads);
      heapifyLevelKernel<<<blocks,threads>>>(dA,n,start,count);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  // 2) Извлекаем максимум по одному, получаем отсортированный массив.
  heapExtractKernel<<<1,1>>>(dA,n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(a.data(),dA,n*sizeof(int),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dA));
}

void cpu_heap_sort(std::vector<int>& a){
  // Последовательная версия на CPU.
  std::make_heap(a.begin(),a.end());
  std::sort_heap(a.begin(),a.end());
}

/* =========================
   Task 4: Benchmark
   ========================= */

struct Res{ double cpu_ms=0; float gpu_ms=0; bool ok_cpu=false; bool ok_gpu=false; };

template<class CPUF, class GPUF>
Res bench(const std::vector<int>& input, CPUF cpuF, GPUF gpuF){
  Res r;

  { // CPU: измеряем chrono
    auto a=input;
    auto t0=std::chrono::high_resolution_clock::now();
    cpuF(a);
    auto t1=std::chrono::high_resolution_clock::now();
    r.cpu_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    r.ok_cpu=is_sorted_cpu(a);
  }

  { // GPU: измеряем cudaEvent
    auto a=input;
    cudaEvent_t s,t;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&t));
    CUDA_CHECK(cudaEventRecord(s));
    gpuF(a);
    CUDA_CHECK(cudaEventRecord(t));
    CUDA_CHECK(cudaEventSynchronize(t));
    r.gpu_ms=elapsed_ms(s,t);
    r.ok_gpu=is_sorted_cpu(a);
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(t));
  }

  return r;
}

int main(){
  std::cout<<std::fixed<<std::setprecision(3);

  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(-1000000,1000000);

  std::vector<int> sizes={10000,100000,1000000};

  std::cout<<"1) CUDA merge sort + CPU version\n";
  std::cout<<"2) CUDA quick sort + CPU version\n";
  std::cout<<"3) CUDA heap sort + CPU version\n";
  std::cout<<"4) Performance comparison\n\n";

  for(int n: sizes){
    std::vector<int> input(n);
    for(int i=0;i<n;++i) input[i]=dist(rng);

    std::cout<<"----------------------------------------\n";
    std::cout<<"Array size: "<<n<<"\n\n";

    { // 1) Merge
      auto r=bench(input,cpu_merge_sort,gpu_merge_sort);
      std::cout<<"1) Merge sort\n";
      std::cout<<"   CPU time (ms): "<<r.cpu_ms<<" | sorted: "<<(r.ok_cpu?"YES":"NO")<<"\n";
      std::cout<<"   GPU time (ms): "<<r.gpu_ms<<" | sorted: "<<(r.ok_gpu?"YES":"NO")<<"\n";
      if(r.gpu_ms>0) std::cout<<"   Speedup (CPU/GPU): "<<(r.cpu_ms/r.gpu_ms)<<"x\n";
      std::cout<<"\n";
    }

    { // 2) Quick
      auto r=bench(input,cpu_quick_sort,gpu_quick_sort);
      std::cout<<"2) Quick sort\n";
      std::cout<<"   CPU time (ms): "<<r.cpu_ms<<" | sorted: "<<(r.ok_cpu?"YES":"NO")<<"\n";
      std::cout<<"   GPU time (ms): "<<r.gpu_ms<<" | sorted: "<<(r.ok_gpu?"YES":"NO")<<"\n";
      if(r.gpu_ms>0) std::cout<<"   Speedup (CPU/GPU): "<<(r.cpu_ms/r.gpu_ms)<<"x\n";
      std::cout<<"\n";
    }

    { // 3) Heap
      auto r=bench(input,cpu_heap_sort,gpu_heap_sort);
      std::cout<<"3) Heap sort\n";
      std::cout<<"   CPU time (ms): "<<r.cpu_ms<<" | sorted: "<<(r.ok_cpu?"YES":"NO")<<"\n";
      std::cout<<"   GPU time (ms): "<<r.gpu_ms<<" | sorted: "<<(r.ok_gpu?"YES":"NO")<<"\n";
      if(r.gpu_ms>0) std::cout<<"   Speedup (CPU/GPU): "<<(r.cpu_ms/r.gpu_ms)<<"x\n";
      std::cout<<"\n";
    }

    std::cout<<"4) Conclusion\n";
    std::cout<<"   Large arrays benefit from parallel work; parts with inherent sequential steps limit speedup.\n\n";
  }

  return 0;
}
