# Matrix Multiplication CUDA Explanation

## Problem Overview

The task is to implement a CUDA kernel to perform matrix multiplication: $C = A \times B$.

* Matrix $A$ has dimensions $M \times N$.
* Matrix $B$ has dimensions $N \times K$.
* The result, Matrix $C$, has dimensions $M \times K$.

All matrices are stored in **row-major** format in 1D arrays.

## Key Concepts

### 1. 2D Thread Hierarchy

Unlike vector addition which uses a 1D grid, matrix operations map naturally to a 2D grid of threads. We assign one thread to compute one element of the output matrix $C$.

* **Grid Dimensions**: `blocksPerGrid` is defined as a `dim3` type with `x` and `y` components.
* **Block Dimensions**: `threadsPerBlock` is also 2D (e.g., $16 \times 16$).

### 2. Global Index Calculation (2D)

We map the thread coordinates to the matrix row and column indices:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

* **Row (y-dimension)**: Corresponds to the $M$ dimension of Matrix $C$.
* **Column (x-dimension)**: Corresponds to the $K$ dimension of Matrix $C$.

### 3. Boundary Check

Since the matrix dimensions ($M, K$) might not be multiples of the block dimensions ($16, 16$), we must ensure the thread is within the valid range of the output matrix:

```cpp
if (row < M && col < K) {
    // Compute element C[row, col]
}
```

### 4. Dot Product Calculation & Linear Indexing

Each element $C_{row, col}$ is the dot product of the $row$-th row of $A$ and the $col$-th column of $B$.
Since matrices are stored in row-major order (1D arrays), we access elements using linear indices:

* Element at `A[row][i]` is at index `row * N + i`.
* Element at `B[i][col]` is at index `i * K + col`.
* Element at `C[row][col]` is at index `row * K + col`.

```cpp
float sum = 0.0f;
for (int i = 0; i < N; ++i) {
    sum += A[row * N + i] * B[i * K + col];
}
C[row * K + col] = sum;
```

### 5. Kernel Launch Configuration

The host code configures the 2D grid:

```cpp
dim3 threadsPerBlock(16, 16);
dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
```

* The `x` dimension of the grid covers the columns ($K$).
* The `y` dimension of the grid covers the rows ($M$).
* Ceiling division ensures full coverage of the matrix.
