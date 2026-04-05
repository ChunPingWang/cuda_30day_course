#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Day 14: 矩陣乘法週末專題
 *
 * 比較三種實作方式：
 * 1. CPU 版本
 * 2. GPU 基本版本
 * 3. GPU 共享記憶體優化版本
 */

#define TILE_SIZE 16

// 錯誤檢查巨集：每個 CUDA API 呼叫都應包裝此巨集
// 💡 Debug 提示：如果程式莫名結束，檢查是否有 CUDA API 回傳錯誤
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CPU 矩陣乘法（三層迴圈，作為 GPU 結果的驗證基準）
void matmulCPU(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {        // 遍歷結果矩陣的每一列
        for (int j = 0; j < N; j++) {    // 遍歷結果矩陣的每一行
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {  // 內積運算
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// GPU 基本版本：每個執行緒計算 C 矩陣的一個元素，直接從全域記憶體讀取
__global__ void matmulBasic(float *A, float *B, float *C, int M, int N, int K) {
    // 2D 索引計算：blockIdx/threadIdx 的 .y 對應 row，.x 對應 col
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // ⚠️ 注意：每次迴圈都存取全域記憶體，頻寬是瓶頸
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// GPU 共享記憶體優化版本：將資料分塊載入快速的共享記憶體再計算
__global__ void matmulTiled(float *A, float *B, float *C, int M, int N, int K) {
    // __shared__ 共享記憶體：同 Block 內所有執行緒共享，比全域記憶體快約 100 倍
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍歷所有 tiles（將 K 維度切成多個 TILE_SIZE 的區塊）
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 每個執行緒協作載入一個元素到共享記憶體
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        if (row < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;  // 超出邊界填 0

        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // ⚠️ 注意：必須同步！確保 tile 完全載入後才能開始計算
        __syncthreads();

        // 用共享記憶體中的 tile 做乘加（存取速度快很多）
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // 同步：確保所有執行緒算完才載入下一個 tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 驗證結果
bool verifyResult(float *A, float *B, int size) {
    for (int i = 0; i < size; i++) {
        if (fabsf(A[i] - B[i]) > 1e-3) {
            printf("Mismatch at %d: CPU=%.4f, GPU=%.4f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("========================================\n");
    printf("    矩陣乘法效能比較\n");
    printf("========================================\n\n");

    // 矩陣大小
    int sizes[] = {256, 512, 1024};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < numSizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);

        printf("矩陣大小: %d x %d\n", M, N);
        printf("----------------------------------------\n");

        // 分配主機記憶體
        float *h_A = (float*)malloc(sizeA);
        float *h_B = (float*)malloc(sizeB);
        float *h_C_cpu = (float*)malloc(sizeC);
        float *h_C_basic = (float*)malloc(sizeC);
        float *h_C_tiled = (float*)malloc(sizeC);

        // 初始化矩陣
        for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

        // 分配 GPU 記憶體並用 CHECK_CUDA 檢查錯誤
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, sizeA));  // 在 GPU 上分配矩陣空間
        CHECK_CUDA(cudaMalloc(&d_B, sizeB));
        CHECK_CUDA(cudaMalloc(&d_C, sizeC));

        // 將資料從 CPU（Host）複製到 GPU（Device）
        CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // CPU 測試
        clock_t cpuStart = clock();
        matmulCPU(h_A, h_B, h_C_cpu, M, N, K);
        clock_t cpuEnd = clock();
        float cpuTime = (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC * 1000;
        printf("CPU:              %8.2f ms\n", cpuTime);

        // GPU 基本版本
        // dim3 設定 2D 的 Block 和 Grid 維度
        dim3 threads(TILE_SIZE, TILE_SIZE);  // 每個 Block 16×16 = 256 個執行緒
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        // 暖機（Warm-up）：第一次執行較慢（GPU 初始化），計時前先跑一次
        matmulBasic<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        matmulBasic<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float basicTime;
        cudaEventElapsedTime(&basicTime, start, stop);
        CHECK_CUDA(cudaMemcpy(h_C_basic, d_C, sizeC, cudaMemcpyDeviceToHost));
        printf("GPU 基本版:       %8.2f ms (%.1fx)\n", basicTime, cpuTime / basicTime);

        // GPU 共享記憶體版本
        matmulTiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        matmulTiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float tiledTime;
        cudaEventElapsedTime(&tiledTime, start, stop);
        CHECK_CUDA(cudaMemcpy(h_C_tiled, d_C, sizeC, cudaMemcpyDeviceToHost));
        printf("GPU 共享記憶體:   %8.2f ms (%.1fx)\n", tiledTime, cpuTime / tiledTime);

        // 驗證
        bool basicOK = verifyResult(h_C_cpu, h_C_basic, M * N);
        bool tiledOK = verifyResult(h_C_cpu, h_C_tiled, M * N);
        printf("驗證: 基本版=%s, 共享記憶體版=%s\n",
               basicOK ? "通過" : "失敗", tiledOK ? "通過" : "失敗");
        printf("\n");

        // 清理：釋放 CPU 和 GPU 記憶體
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_basic); free(h_C_tiled);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);  // 釋放 GPU 記憶體
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("========================================\n");
    printf("結論：\n");
    printf("1. GPU 比 CPU 快很多倍\n");
    printf("2. 共享記憶體優化可以進一步提升效能\n");
    printf("3. 矩陣越大，加速效果越明顯\n");
    printf("========================================\n");

    return 0;
}
