#include <stdio.h>
#include <stdlib.h>

/**
 * Day 15: 記憶體合併（Memory Coalescing）範例
 *
 * 展示記憶體存取模式對效能的影響
 */

#define N (1 << 22)  // 4M 元素
#define BLOCK_SIZE 256

// 合併存取：連續執行緒存取連續記憶體
// __global__ 表示這是一個「核心函式」，由 GPU 上的多個執行緒同時執行
__global__ void coalescedAccess(float *input, float *output, int n) {
    // 計算此執行緒的全域索引：blockIdx.x=第幾個區塊, blockDim.x=每個區塊有幾個執行緒, threadIdx.x=區塊內第幾個執行緒
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ⚠️ 注意：一定要做邊界檢查，否則會存取超出陣列範圍的記憶體
    if (idx < n) {
        // 連續的執行緒存取連續的記憶體位址 → GPU 可以合併成一次記憶體交易，效率最高
        output[idx] = input[idx] * 2.0f;
    }
}

// 跨步存取：執行緒跳躍存取（效率低）
// 💡 Debug 提示：如果你的 kernel 效能很差，檢查是否有跨步存取的情況
__global__ void stridedAccess(float *input, float *output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 每個執行緒跳過 stride 個元素存取 → 記憶體不連續，GPU 無法合併，效能大幅下降
    int strided_idx = (idx * stride) % n;
    if (idx < n / stride) {
        output[strided_idx] = input[strided_idx] * 2.0f;
    }
}

// 結構陣列（AoS, Array of Structures）- 非最佳存取模式
// 記憶體排列：[x0,y0,z0,vx0,vy0,vz0, x1,y1,z1,vx1,vy1,vz1, ...]
// ⚠️ 注意：相鄰執行緒存取的 x 值在記憶體中隔了 6 個 float，無法合併存取
struct Particle_AoS {
    float x, y, z;
    float vx, vy, vz;
};

// 用 AoS 方式更新粒子位置（效率較低）
__global__ void processAoS(Particle_AoS *particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 每個執行緒存取的資料不連續（thread0 讀 x0, thread1 讀 x1，但它們在記憶體中相隔 24 bytes）
        particles[idx].x += particles[idx].vx;
        particles[idx].y += particles[idx].vy;
        particles[idx].z += particles[idx].vz;
    }
}

// 陣列結構（SoA, Structure of Arrays）- 最佳存取模式
// 記憶體排列：x 陣列=[x0,x1,x2,...], y 陣列=[y0,y1,y2,...], ...
// 相鄰執行緒存取相鄰記憶體 → 完美合併存取！
struct Particles_SoA {
    float *x, *y, *z;
    float *vx, *vy, *vz;
};

// 用 SoA 方式更新粒子位置（效率最高）
__global__ void processSoA(float *x, float *y, float *z,
                           float *vx, float *vy, float *vz, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 每個陣列都是連續存取（thread0 讀 x[0], thread1 讀 x[1]，記憶體連續）
        x[idx] += vx[idx];
        y[idx] += vy[idx];
        z[idx] += vz[idx];
    }
}

// 計時輔助函式：測量某個 kernel 執行 100 次的平均耗時
float benchmarkKernel(void (*setup)(void), void (*kernel)(void), const char *name) {
    cudaEvent_t start, stop;  // CUDA 事件，用於精確計時
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 暖機：先跑一次讓 GPU 暖機（第一次執行通常較慢，因為有初始化開銷）
    if (setup) setup();
    kernel();
    cudaDeviceSynchronize();  // 等待 GPU 完成所有工作

    // 計時（執行多次取平均）
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 100.0f;
}

// 全域變數（用於 benchmark）
float *d_input, *d_output;
int g_n;

void runCoalesced() {
    // <<<grid大小, block大小>>> 是 CUDA 啟動核心函式的語法
    // (g_n + BLOCK_SIZE - 1) / BLOCK_SIZE 是「無條件進位除法」，確保有足夠的區塊涵蓋所有元素
    coalescedAccess<<<(g_n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output, g_n);
}

void runStrided() {
    stridedAccess<<<(g_n / 32 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output, g_n, 32);
}

int main() {
    printf("========================================\n");
    printf("    記憶體合併效能比較\n");
    printf("========================================\n\n");

    g_n = N;
    size_t size = N * sizeof(float);

    // 分配主機（CPU）記憶體
    float *h_input = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }

    cudaMalloc(&d_input, size);   // 在 GPU 上分配記憶體（d_ 前綴代表 device）
    cudaMalloc(&d_output, size);  // 💡 Debug 提示：cudaMalloc 失敗時回傳錯誤碼，正式程式應檢查回傳值
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);  // 將資料從 CPU 複製到 GPU

    printf("資料大小: %d 個元素 (%.1f MB)\n\n", N, size / (1024.0f * 1024.0f));

    // 測試 1：合併 vs 跨步存取
    printf("【測試 1: 存取模式比較】\n");
    float timeCoalesced = benchmarkKernel(NULL, runCoalesced, "合併存取");
    float timeStrided = benchmarkKernel(NULL, runStrided, "跨步存取");

    printf("合併存取:   %.3f ms\n", timeCoalesced);
    printf("跨步存取:   %.3f ms\n", timeStrided);
    printf("效能差異:   %.2fx\n\n", timeStrided / timeCoalesced);

    // 測試 2：AoS vs SoA
    printf("【測試 2: AoS vs SoA】\n");

    int numParticles = 1 << 20;  // 1M 粒子

    // AoS
    Particle_AoS *d_particles_aos;
    cudaMalloc(&d_particles_aos, numParticles * sizeof(Particle_AoS));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 暖機
    processAoS<<<(numParticles + 255) / 256, 256>>>(d_particles_aos, numParticles);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        processAoS<<<(numParticles + 255) / 256, 256>>>(d_particles_aos, numParticles);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeAoS;
    cudaEventElapsedTime(&timeAoS, start, stop);
    timeAoS /= 100.0f;

    // SoA
    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    cudaMalloc(&d_x, numParticles * sizeof(float));
    cudaMalloc(&d_y, numParticles * sizeof(float));
    cudaMalloc(&d_z, numParticles * sizeof(float));
    cudaMalloc(&d_vx, numParticles * sizeof(float));
    cudaMalloc(&d_vy, numParticles * sizeof(float));
    cudaMalloc(&d_vz, numParticles * sizeof(float));

    // 暖機
    processSoA<<<(numParticles + 255) / 256, 256>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, numParticles);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        processSoA<<<(numParticles + 255) / 256, 256>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, numParticles);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeSoA;
    cudaEventElapsedTime(&timeSoA, start, stop);
    timeSoA /= 100.0f;

    printf("AoS (結構陣列): %.3f ms\n", timeAoS);
    printf("SoA (陣列結構): %.3f ms\n", timeSoA);
    printf("效能差異:       %.2fx\n\n", timeAoS / timeSoA);

    // 總結
    printf("========================================\n");
    printf("    總結\n");
    printf("========================================\n");
    printf("1. 記憶體合併存取可以顯著提升效能\n");
    printf("2. 連續執行緒應該存取連續的記憶體位址\n");
    printf("3. 使用 SoA 比 AoS 更適合 GPU\n");
    printf("4. 避免跨步存取，特別是大跨步\n");

    // 清理：釋放所有分配的記憶體
    free(h_input);        // 釋放 CPU 記憶體
    cudaFree(d_input);    // 釋放 GPU 記憶體
    cudaFree(d_output);   // ⚠️ 注意：忘記 cudaFree 會造成 GPU 記憶體洩漏
    cudaFree(d_particles_aos);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
