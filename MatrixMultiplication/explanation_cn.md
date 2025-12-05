# 矩阵乘法 (Matrix Multiplication) CUDA 详解

## 题目概述
本题要求实现一个 CUDA kernel 来执行矩阵乘法：$C = A \times B$。
* 矩阵 $A$ 的维度为 $M \times N$。
* 矩阵 $B$ 的维度为 $N \times K$。
* 结果矩阵 $C$ 的维度为 $M \times K$。

所有矩阵都以 **行主序 (row-major)** 格式存储在一维数组中。

## 核心知识点

### 1. 二维线程模型 (2D Thread Hierarchy)
与使用一维网格的向量加法不同，矩阵运算自然地映射到二维线程网格。我们分配一个线程来计算输出矩阵 $C$ 中的一个元素。

* **Grid 维度**: `blocksPerGrid` 定义为 `dim3` 类型，具有 `x` 和 `y` 分量。
* **Block 维度**: `threadsPerBlock` 也是二维的（例如 $16 \times 16$）。

### 2. 全局索引计算 (2D) (Global Index Calculation)
我们将线程坐标映射到矩阵的行和列索引：

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

* **行 (y 维)**: 对应矩阵 $C$ 的 $M$ 维度。
* **列 (x 维)**: 对应矩阵 $C$ 的 $K$ 维度。

### 3. 边界检查 (Boundary Check)
由于矩阵维度 ($M, K$) 可能不是块维度 ($16, 16$) 的整数倍，我们需要确保线程处于输出矩阵的有效范围内：

```cpp
if (row < M && col < K) {
    // 计算元素 C[row, col]
}
```

### 4. 点积计算与线性索引 (Dot Product Calculation & Linear Indexing)
元素 $C_{row, col}$ 是 $A$ 的第 $row$ 行与 $B$ 的第 $col$ 列的点积。
由于矩阵以行主序（一维数组）存储，我们使用线性索引访问元素：

* `A[row][i]` 位于索引 `row * N + i`。
* `B[i][col]` 位于索引 `i * K + col`。
* `C[row][col]` 位于索引 `row * K + col`。

```cpp
float sum = 0.0f;
for (int i = 0; i < N; ++i) {
    sum += A[row * N + i] * B[i * K + col];
}
C[row * K + col] = sum;
```

### 5. Kernel 启动配置 (Kernel Launch Configuration)
Host 代码配置二维网格：

```cpp
dim3 threadsPerBlock(16, 16);
dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
```

* 网格的 `x` 维度覆盖列 ($K$)。
* 网格的 `y` 维度覆盖行 ($M$)。
* 向上取整除法确保完全覆盖矩阵。

```