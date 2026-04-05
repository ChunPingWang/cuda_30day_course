#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

/**
 * 矩陣乘法：基本版本（每個執行緒計算結果矩陣 C 的一個元素）
 * C = A × B
 * A: M×K, B: K×N, C: M×N
 */
// __global__ 核心函式，由 CPU 呼叫、GPU 執行
__global__ void matrixMulBasic(float *A, float *B, float *C,
                               int M, int K, int N) {
    // 使用 2D 的 Block/Thread 配置來對應矩陣的行與列
    // blockIdx.y/threadIdx.y 對應列（row），blockIdx.x/threadIdx.x 對應行（col）
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {  // ⚠️ 注意：2D 邊界檢查，row 和 col 都要檢查
        float sum = 0.0f;

        // 計算 C[row][col] = Σ A[row][k] * B[k][col]（k 從 0 到 K-1）
        for (int k = 0; k < K; k++) {
            // 1D 陣列模擬 2D 矩陣：mat[i][j] = mat[i * 列數 + j]
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

/**
 * CPU 版本（用於驗證）
 */
void matrixMulCPU(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void initMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 10);
    }
}

void printMatrix(float *mat, int rows, int cols, const char *name) {
    printf("%s (%d×%d):\n", name, rows, cols);
    for (int i = 0; i < rows && i < 4; i++) {  // 只印前 4 行
        printf("  ");
        for (int j = 0; j < cols && j < 4; j++) {  // 只印前 4 列
            printf("%6.1f ", mat[i * cols + j]);
        }
        if (cols > 4) printf("...");
        printf("\n");
    }
    if (rows > 4) printf("  ...\n");
    printf("\n");
}

bool verify(float *gpu, float *cpu, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(gpu[i] - cpu[i]) > 0.1f) {
            printf("Mismatch at %d: GPU=%.2f, CPU=%.2f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("========================================\n");
    printf("    矩陣乘法：基本版本\n");
    printf("========================================\n\n");

    // 矩陣維度
    int M = 512;  // A 的行數
    int K = 256;  // A 的列數 = B 的行數
    int N = 512;  // B 的列數

    printf("矩陣維度: A(%d×%d) × B(%d×%d) = C(%d×%d)\n\n", M, K, K, N, M, N);

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // 分配主機記憶體
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);
    float *h_C_ref = (float*)malloc(bytesC);

    // 初始化
    srand(42);
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // 分配 GPU（設備）記憶體
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);  // 在 GPU 上分配矩陣 A 的空間
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    // 將矩陣資料從 CPU 複製到 GPU
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    // 設定 2D 執行配置（矩陣運算適合用 2D Grid/Block）
    // dim3 是 CUDA 的三維向量類型，用來指定 Grid 和 Block 的維度
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 每個 Block 有 16×16 = 256 個執行緒
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,   // x 方向的 Block 數（對應 col）
                   (M + BLOCK_SIZE - 1) / BLOCK_SIZE);  // y 方向的 Block 數（對應 row）

    printf("執行配置:\n");
    printf("  Block: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("  Grid:  (%d, %d)\n", numBlocks.x, numBlocks.y);
    printf("  總執行緒數: %d\n\n", numBlocks.x * numBlocks.y *
                                   threadsPerBlock.x * threadsPerBlock.y);

    // 計時
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // GPU 執行
    cudaEventRecord(start);
    // <<<numBlocks, threadsPerBlock>>> 是 CUDA 的核心函式啟動語法
    // numBlocks = Grid 大小（幾個 Block），threadsPerBlock = Block 大小（幾個執行緒）
    matrixMulBasic<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    // 💡 Debug 提示：核心啟動是非同步的，如果結果不對，先用 cudaGetLastError() 檢查啟動是否成功
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // 將 GPU 計算結果複製回 CPU（cudaMemcpyDeviceToHost = GPU → CPU）
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    // CPU 執行（用於驗證）
    printf("驗證中（CPU 計算）...\n");
    clock_t cpuStart = clock();
    matrixMulCPU(h_A, h_B, h_C_ref, M, K, N);
    clock_t cpuEnd = clock();
    float cpuTime = ((float)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000;

    // 驗證
    bool correct = verify(h_C, h_C_ref, M * N);

    // 輸出結果
    printf("\n結果:\n");
    printf("  GPU 時間: %.3f ms\n", gpuTime);
    printf("  CPU 時間: %.3f ms\n", cpuTime);
    printf("  加速比:   %.2fx\n", cpuTime / gpuTime);
    printf("  驗證:     %s\n\n", correct ? " 正確" : " 錯誤");

    // 顯示部分矩陣
    printMatrix(h_A, M, K, "A");
    printMatrix(h_B, K, N, "B");
    printMatrix(h_C, M, N, "C = A × B");

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("========================================\n");
    printf("💡 這是未優化的版本\n");
    printf("   明天學習共享記憶體後，效能可提升 5-10 倍！\n");
    printf("========================================\n");

    return 0;
}
