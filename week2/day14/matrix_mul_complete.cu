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

// 錯誤檢查
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CPU 矩陣乘法
void matmulCPU(float *A, float *B, float *C, int M, int N, int K) {
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

// GPU 基本版本
__global__ void matmulBasic(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// GPU 共享記憶體優化版本
__global__ void matmulTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍歷所有 tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 載入 tile 到共享記憶體
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        if (row < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 計算 tile 的部分乘積
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

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

        // 分配 GPU 記憶體
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, sizeA));
        CHECK_CUDA(cudaMalloc(&d_B, sizeB));
        CHECK_CUDA(cudaMalloc(&d_C, sizeC));

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
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        // 暖機
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

        // 清理
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_basic); free(h_C_tiled);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
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
