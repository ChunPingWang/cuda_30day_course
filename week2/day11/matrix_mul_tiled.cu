#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

/**
 * 矩陣乘法：使用共享記憶體（Tiled）
 * 核心概念：將大矩陣分成小塊（tile），先載入到快速的共享記憶體中再計算
 * 共享記憶體比全域記憶體快約 100 倍，可大幅減少記憶體存取延遲
 */
__global__ void matrixMulTiled(float *A, float *B, float *C,
                               int M, int K, int N) {
    // __shared__ 宣告共享記憶體：同一 Block 內所有執行緒共享，速度極快
    // ⚠️ 注意：共享記憶體大小有限（通常 48KB），TILE_SIZE 不能設太大
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // 此執行緒負責的結果矩陣的 row
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // 此執行緒負責的結果矩陣的 col

    float sum = 0.0f;

    // 計算需要多少個 tile 來覆蓋 K 維度
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 步驟 1：每個執行緒協作載入一個元素到共享記憶體（A 的 tile）
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;  // 超出邊界填 0
        }

        // 步驟 2：載入 B 的 tile
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // __syncthreads() 同步：確保 tile 完全載入後才開始計算
        // ⚠️ 注意：如果漏掉這行，有些執行緒可能讀到尚未載入的資料，導致結果錯誤
        __syncthreads();

        // 步驟 3：用共享記憶體中的 tile 做乘加（這裡存取速度快很多）
        #pragma unroll  // 編譯器提示：展開迴圈以提升效能
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // 再次同步：確保所有執行緒都算完，才能安全地載入下一個 tile
        __syncthreads();
    }

    // 寫入結果到全域記憶體
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * 基本版本（用於比較，不使用共享記憶體）
 * 每次迴圈都從全域記憶體讀取，速度較慢
 */
__global__ void matrixMulBasic(float *A, float *B, float *C,
                               int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // 每次迭代都要讀取全域記憶體（慢），這就是 Tiled 版本要優化的地方
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void initMatrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (float)(rand() % 10);
    }
}

bool verify(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(a[i] - b[i]) > 0.1f) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("========================================\n");
    printf("    矩陣乘法：共享記憶體優化比較\n");
    printf("========================================\n\n");

    // 測試不同大小
    int sizes[] = {256, 512, 1024};
    int numSizes = 3;

    for (int s = 0; s < numSizes; s++) {
        int M = sizes[s];
        int K = sizes[s];
        int N = sizes[s];

        printf("矩陣大小: %d × %d\n", M, N);
        printf("Tile 大小: %d × %d\n", TILE_SIZE, TILE_SIZE);
        printf("----------------------------------------\n");

        size_t bytesA = M * K * sizeof(float);
        size_t bytesB = K * N * sizeof(float);
        size_t bytesC = M * N * sizeof(float);

        // 分配記憶體
        float *h_A = (float*)malloc(bytesA);
        float *h_B = (float*)malloc(bytesB);
        float *h_C_basic = (float*)malloc(bytesC);
        float *h_C_tiled = (float*)malloc(bytesC);

        // 分配 GPU 記憶體
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytesA);  // 在 GPU 上分配空間
        cudaMalloc(&d_B, bytesB);
        cudaMalloc(&d_C, bytesC);

        // 初始化
        srand(42);
        initMatrix(h_A, M * K);
        initMatrix(h_B, K * N);

        // 將資料從 CPU 複製到 GPU
        cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

        // 2D 執行配置：Block 大小 = TILE_SIZE × TILE_SIZE
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        // 基本版本
        cudaEventRecord(start);
        matrixMulBasic<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        float basicTime = ms;

        cudaMemcpy(h_C_basic, d_C, bytesC, cudaMemcpyDeviceToHost);

        // Tiled 版本
        cudaEventRecord(start);
        matrixMulTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        float tiledTime = ms;

        cudaMemcpy(h_C_tiled, d_C, bytesC, cudaMemcpyDeviceToHost);

        // 驗證
        bool correct = verify(h_C_basic, h_C_tiled, M * N);

        // 計算 GFLOPS（每秒十億次浮點運算）
        float operations = 2.0f * M * N * K;  // 每個 C 元素需要 K 次乘法 + K 次加法
        float basicGFlops = operations / (basicTime * 1e6);
        float tiledGFlops = operations / (tiledTime * 1e6);

        printf("基本版本: %.3f ms (%.2f GFLOPS)\n", basicTime, basicGFlops);
        printf("Tiled:    %.3f ms (%.2f GFLOPS)\n", tiledTime, tiledGFlops);
        printf("加速比:   %.2fx\n", basicTime / tiledTime);
        printf("驗證:     %s\n\n", correct ? " 正確" : " 錯誤");

        // 清理
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C_basic);
        free(h_C_tiled);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("========================================\n");
    printf("共享記憶體優化效果：\n");
    printf("1. 減少全域記憶體存取\n");
    printf("2. 資料重用提高效能\n");
    printf("3. 矩陣越大，加速比越明顯\n");
    printf("========================================\n");

    return 0;
}
