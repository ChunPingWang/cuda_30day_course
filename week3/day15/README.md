# Day 15: 記憶體合併（Memory Coalescing）深入

## 📚 今日學習目標

- 深入理解記憶體合併的原理
- 學習全域記憶體的存取模式
- 識別並優化非合併存取
- 使用工具分析記憶體效率

## 🧠 全域記憶體架構

### 記憶體交易（Memory Transaction）

GPU 存取全域記憶體的最小單位是 **32 bytes** 或 **128 bytes**。

```
一次記憶體交易可以服務多個執行緒：
┌────────────────────────────────────────┐
│  32 bytes (8 個 float) 或             │
│  128 bytes (32 個 float)               │
└────────────────────────────────────────┘
```

### 合併存取

當一個 Warp（32 執行緒）存取連續的記憶體位置時：

```
Thread 0  → data[0]   ┐
Thread 1  → data[1]   │
Thread 2  → data[2]   │ 合併為 1 次交易
...                   │
Thread 31 → data[31]  ┘
```

### 非合併存取

當存取不連續時：

```
Thread 0  → data[0]     → 1 次交易
Thread 1  → data[32]    → 1 次交易
Thread 2  → data[64]    → 1 次交易
...
總共需要 32 次交易！效率低 32 倍！
```

## 📊 存取模式分析

### 模式 1：理想合併 ✅

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx];
```

### 模式 2：跨步存取 ⚠️

```cuda
int idx = threadIdx.x * stride;  // stride > 1
float val = data[idx];
```

stride = 2 時效率降為 50%
stride = 32 時效率降為 ~3%

### 模式 3：隨機存取 ❌

```cuda
int idx = randomIndex[threadIdx.x];
float val = data[idx];
```

最差情況，幾乎無法合併

## 🔧 優化策略

### 1. 調整資料佈局

將 Array of Structures (AoS) 改為 Structure of Arrays (SoA)：

```cuda
// ❌ AoS - 非合併
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};
Particle particles[N];
// 存取 particles[i].x 時，相鄰執行緒存取不連續

// ✅ SoA - 合併
struct Particles {
    float x[N], y[N], z[N];
    float vx[N], vy[N], vz[N];
};
// 存取 x[i] 時，相鄰執行緒存取連續
```

### 2. 使用共享記憶體轉換

```cuda
__global__ void transpose(float *in, float *out, int N) {
    __shared__ float tile[32][33];  // +1 避免 bank conflict

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // 合併讀取
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    // 合併寫入（轉置後）
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}
```

## 🔧 實作練習

### 範例程式

1. **coalescing_demo.cu** - 記憶體合併示範
2. **aos_soa.cu** - AoS vs SoA 比較
3. **transpose_optimized.cu** - 優化的矩陣轉置

## 📝 今日作業

1. ✅ 理解記憶體合併的原理
2. ✅ 識別程式中的非合併存取
3. ✅ 練習 AoS 到 SoA 的轉換

---

**明天我們將學習 Bank Conflict！** 🏦
