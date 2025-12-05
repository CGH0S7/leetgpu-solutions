# 向量加法 (Vector Addition) CUDA 详解

## 题目概述

本题要求实现一个 CUDA kernel，完成两个浮点数向量的逐元素加法：$C = A + B$。输入向量 $A$ 和 $B$ 长度均为 $N$，结果存储在向量 $C$ 中。

## 核心知识点

### 1. CUDA 线程模型 (Thread Hierarchy)

GPU 计算的核心在于并行化。CUDA 将计算任务分配给大量的线程 (Threads)。这些线程被组织成层级结构：

* **Grid (网格)**: 包含所有线程块 (Block)。
* **Block (线程块)**: 包含一组线程。块内的线程可以通过共享内存协作。
* **Thread (线程)**: 执行计算的最小单元。

在代码中：

* `threadIdx.x`: 当前线程在块内的索引。
* `blockIdx.x`: 当前块在网格内的索引。
* `blockDim.x`: 一个块中包含的线程数量。

### 2. 全局索引计算 (Global Index Calculation)

为了让每个线程处理向量中的一个特定元素，我们需要计算该线程在整个网格中的唯一全局索引 (`idx`)：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

* `blockIdx.x * blockDim.x`: 当前块之前所有块的线程总数（偏移量）。
* `+ threadIdx.x`: 当前线程在块内的偏移。

### 3. 边界检查 (Boundary Check)

由于线程块的大小 (`blockDim`) 通常是固定的（例如 256），而向量长度 $N$ 可能不是块大小的整数倍，因此启动的线程总数可能略多于 $N$。
必须检查索引是否越界：

```cpp
if (idx < N) {
    C[idx] = A[idx] + B[idx];
}
```

只有当 `idx` 在有效范围内 `[0, N-1]` 时，才进行加法操作。

### 4. Kernel 启动参数 (Kernel Launch Configuration)

Host 代码 (CPU) 负责配置 Kernel 的启动参数：

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
```

* **Block Size (`threadsPerBlock`)**: 设为 256 是一个通用的选择（通常是 32 的倍数以适应 Warp 调度）。
* **Grid Size (`blocksPerGrid`)**: 计算公式 `(N + threadsPerBlock - 1) / threadsPerBlock` 实现了向上取整的除法，确保即使 $N$ 不能整除块大小，也有足够的线程覆盖所有元素。

### 5. 设备指针 (Device Pointers)

函数 `solve` 接收的指针 `A`, `B`, `C` 指向 GPU 的全局内存 (Global Memory)。Kernel 可以直接通过这些指针读写数据。
