# Vector Addition CUDA Explanation

## Problem Overview
The task is to implement a CUDA kernel to perform element-wise addition of two floating-point vectors: $C = A + B$. The input vectors $A$ and $B$ have length $N$, and the result is stored in vector $C$.

## Key Concepts

### 1. CUDA Thread Hierarchy
GPU computing relies on massive parallelism. CUDA organizes computation into a hierarchy of threads:
*   **Grid**: Contains all thread blocks.
*   **Block**: Contains a group of threads. Threads within a block can cooperate via shared memory.
*   **Thread**: The smallest unit of execution.

In the code:
*   `threadIdx.x`: The index of the current thread within its block.
*   `blockIdx.x`: The index of the current block within the grid.
*   `blockDim.x`: The number of threads in a block.

### 2. Global Index Calculation
To map each thread to a specific element in the vectors, we calculate a unique global index (`idx`) for the thread:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

*   `blockIdx.x * blockDim.x`: The total number of threads in all blocks preceding the current one (the offset).
*   `+ threadIdx.x`: The offset of the current thread within its block.

### 3. Boundary Check
Since the block size (`blockDim`) is fixed (e.g., 256) and the vector length $N$ might not be a perfect multiple of the block size, the total number of launched threads might exceed $N$.
We must guard against out-of-bounds memory access:

```cpp
if (idx < N) {
    C[idx] = A[idx] + B[idx];
}
```
The addition is performed only if `idx` is within the valid range `[0, N-1]`.

### 4. Kernel Launch Configuration
The Host code (CPU) determines the execution configuration:

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
```

*   **Block Size (`threadsPerBlock`)**: 256 is a common choice (should be a multiple of 32 for efficient Warp scheduling).
*   **Grid Size (`blocksPerGrid`)**: The formula `(N + threadsPerBlock - 1) / threadsPerBlock` performs a ceiling division. This ensures that enough thread blocks are launched to cover all $N$ elements, even if $N$ is not divisible by `threadsPerBlock`.

### 5. Device Pointers
The pointers `A`, `B`, and `C` passed to the `solve` function point to Global Memory on the GPU. The kernel can directly read from and write to these locations.
