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
__global__ void coalescedAccess(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

// 跨步存取：執行緒跳躍存取（效率低）
__global__ void stridedAccess(float *input, float *output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (idx * stride) % n;
    if (idx < n / stride) {
        output[strided_idx] = input[strided_idx] * 2.0f;
    }
}

// 結構陣列（AoS）- 非最佳存取模式
struct Particle_AoS {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void processAoS(Particle_AoS *particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 每個執行緒存取的資料不連續
        particles[idx].x += particles[idx].vx;
        particles[idx].y += particles[idx].vy;
        particles[idx].z += particles[idx].vz;
    }
}

// 陣列結構（SoA）- 最佳存取模式
struct Particles_SoA {
    float *x, *y, *z;
    float *vx, *vy, *vz;
};

__global__ void processSoA(float *x, float *y, float *z,
                           float *vx, float *vy, float *vz, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 每個陣列都是連續存取
        x[idx] += vx[idx];
        y[idx] += vy[idx];
        z[idx] += vz[idx];
    }
}

float benchmarkKernel(void (*setup)(void), void (*kernel)(void), const char *name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 暖機
    if (setup) setup();
    kernel();
    cudaDeviceSynchronize();

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

    // 分配記憶體
    float *h_input = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

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

    // 清理
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);
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
