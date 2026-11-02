// GPU CPA records for the coarse search (see src/dopploc/search_gpu.py).
//
// Samples of all flights (time order within a flight, flights back to back) live on the
// device. For every (sample k, lattice cell within rmax) a thread checks whether the 3D
// distance from the cell to the track has a local minimum at k (neighbours k-1, k+1 of the
// same flight), refines it with a parabola through the three squared distances (exact for
// straight flight), filters by distance band, speed band and an optional cell mask, and
// emits the record. Coordinates are made relative to the chunk origin in double, then the
// geometry runs in float32.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define CK(x)                                  \
    do {                                       \
        cudaError_t e_ = (x);                  \
        if (e_ != cudaSuccess) return (int)e_; \
    } while (0)

static int32_t *g_fi = 0;
static double *g_x = 0, *g_y = 0, *g_t = 0;
static float *g_h = 0, *g_gs = 0;
static int64_t g_n = 0;

extern "C" int dl_search_free(void) {
    cudaFree(g_fi); cudaFree(g_x); cudaFree(g_y); cudaFree(g_t); cudaFree(g_h); cudaFree(g_gs);
    g_fi = 0; g_x = g_y = g_t = 0; g_h = g_gs = 0; g_n = 0;
    return 0;
}

extern "C" int dl_search_upload(const int32_t *fi, const double *x, const double *y, const double *t,
                                const float *h, const float *gs, int64_t n, int device) {
    CK(cudaSetDevice(device));
    dl_search_free();
    CK(cudaMalloc(&g_fi, n * sizeof(int32_t)));
    CK(cudaMalloc(&g_x, n * sizeof(double)));
    CK(cudaMalloc(&g_y, n * sizeof(double)));
    CK(cudaMalloc(&g_t, n * sizeof(double)));
    CK(cudaMalloc(&g_h, n * sizeof(float)));
    CK(cudaMalloc(&g_gs, n * sizeof(float)));
    CK(cudaMemcpy(g_fi, fi, n * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_y, y, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_t, t, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_h, h, n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_gs, gs, n * sizeof(float), cudaMemcpyHostToDevice));
    g_n = n;
    return 0;
}

struct Rec {
    int64_t cell;   // ci * ny + cj within the chunk
    int64_t k;
    double tau, trecv;
    float d, gs;
};

__device__ __forceinline__ float dist2(int64_t k, double ox, double oy, float cx, float cy, float ez,
                                       const double *x, const double *y, const float *h) {
    float dx = (float)(x[k] - ox) - cx;
    float dy = (float)(y[k] - oy) - cy;
    float dz = h[k] - ez;
    return dx * dx + dy * dy + dz * dz;
}

__global__ void k_cpa(const int64_t *sidx, int64_t m, int B, const int32_t *fi, const double *x, const double *y,
                      const double *t, const float *h, const float *gs, int64_t n, const float *elev, int nx, int ny,
                      double ox, double oy, float g, const uint8_t *mask, float r2, float d2lo, float d2hi, float vlo,
                      float vhi, double c, Rec *out, unsigned long long *count, int64_t cap) {
    const int64_t W = 2 * B + 1;
    const int64_t total = m * W * W;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t q = idx / (W * W);
        int64_t o = idx % (W * W);
        int64_t k = sidx[q];
        if (k <= 0 || k >= n - 1 || fi[k - 1] != fi[k] || fi[k + 1] != fi[k]) continue;
        float px = (float)(x[k] - ox), py = (float)(y[k] - oy);
        int ci = (int)floorf(px / g) + (int)(o / W) - B;
        int cj = (int)floorf(py / g) + (int)(o % W) - B;
        if (ci < 0 || cj < 0 || ci >= nx || cj >= ny) continue;
        int64_t cell = (int64_t)ci * ny + cj;
        if (mask && !mask[cell]) continue;
        float cx = (ci + 0.5f) * g, cy = (cj + 0.5f) * g;
        float hx = px - cx, hy = py - cy;
        if (hx * hx + hy * hy > r2) continue;
        float ez = elev[cell];
        float dm = dist2(k, ox, oy, cx, cy, ez, x, y, h);
        float db = dist2(k - 1, ox, oy, cx, cy, ez, x, y, h);
        if (!(dm < db)) continue;
        float da = dist2(k + 1, ox, oy, cx, cy, ez, x, y, h);
        if (!(dm <= da)) continue;
        float den = db - 2.0f * dm + da, off = 0.0f, d2 = dm;
        if (den > 0.0f) {
            off = fminf(1.0f, fmaxf(-1.0f, 0.5f * (db - da) / den));
            d2 = dm - 0.25f * (db - da) * off;
        }
        if (d2 < d2lo || d2 > d2hi) continue;
        int64_t nb = off > 0 ? k + 1 : k - 1;
        float s = gs[k] + fabsf(off) * (gs[nb] - gs[k]);
        if (s < vlo || s > vhi) continue;
        double tau = t[k] + (double)off * (t[off > 0 ? k + 1 : k] - t[off > 0 ? k : k - 1]);
        float d = sqrtf(fmaxf(d2, 0.0f));
        unsigned long long slot = atomicAdd(count, 1ULL);
        if ((int64_t)slot < cap) {
            Rec r;
            r.cell = cell;
            r.k = k;
            r.tau = tau;
            r.trecv = tau + d / c;
            r.d = d;
            r.gs = s;
            out[slot] = r;
        }
    }
}

// Records for one chunk. Returns 0 and the number of records in *n_out (which may exceed
// cap: then only cap records were written and the caller must retry with a larger cap).
extern "C" int dl_search_chunk(const int64_t *sidx, int64_t m, const float *elev, int32_t nx, int32_t ny,
                               double ox, double oy, double g, const uint8_t *mask, double rmax, double dlo,
                               double dhi, double vlo, double vhi, double c, int64_t cap, int64_t *out_cell,
                               int64_t *out_k, double *out_tau, double *out_trecv, float *out_d, float *out_gs,
                               int64_t *n_out) {
    if (!g_n) return -1;
    int64_t *dsidx = 0;
    float *delev = 0;
    uint8_t *dmask = 0;
    Rec *dout = 0;
    unsigned long long *dcount = 0, hcount = 0;
    CK(cudaMalloc(&dsidx, m * sizeof(int64_t) + 8));
    CK(cudaMalloc(&delev, (size_t)nx * ny * sizeof(float)));
    CK(cudaMalloc(&dout, cap * sizeof(Rec) + 8));
    CK(cudaMalloc(&dcount, sizeof(unsigned long long)));
    CK(cudaMemcpy(dsidx, sidx, m * sizeof(int64_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(delev, elev, (size_t)nx * ny * sizeof(float), cudaMemcpyHostToDevice));
    if (mask) {
        CK(cudaMalloc(&dmask, (size_t)nx * ny));
        CK(cudaMemcpy(dmask, mask, (size_t)nx * ny, cudaMemcpyHostToDevice));
    }
    CK(cudaMemset(dcount, 0, sizeof(unsigned long long)));
    int B = (int)ceil(rmax / g) + 1;
    int64_t total = m * (int64_t)(2 * B + 1) * (2 * B + 1);
    int blocks = (int)((total + 255) / 256);
    if (blocks > 65535 * 64) blocks = 65535 * 64;
    if (blocks < 1) blocks = 1;
    k_cpa<<<blocks, 256>>>(dsidx, m, B, g_fi, g_x, g_y, g_t, g_h, g_gs, g_n, delev, nx, ny, ox, oy, (float)g, dmask,
                           (float)(rmax * rmax), (float)(dlo * dlo), (float)(dhi * dhi), (float)vlo, (float)vhi, c,
                           dout, dcount, cap);
    CK(cudaGetLastError());
    CK(cudaMemcpy(&hcount, dcount, sizeof(hcount), cudaMemcpyDeviceToHost));
    int64_t nrec = (int64_t)hcount < cap ? (int64_t)hcount : cap;
    if (nrec > 0) {
        Rec *h = (Rec *)malloc(nrec * sizeof(Rec));
        CK(cudaMemcpy(h, dout, nrec * sizeof(Rec), cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < nrec; i++) {
            out_cell[i] = h[i].cell;
            out_k[i] = h[i].k;
            out_tau[i] = h[i].tau;
            out_trecv[i] = h[i].trecv;
            out_d[i] = h[i].d;
            out_gs[i] = h[i].gs;
        }
        free(h);
    }
    *n_out = (int64_t)hcount;
    cudaFree(dsidx); cudaFree(delev); cudaFree(dout); cudaFree(dcount); cudaFree(dmask);
    return 0;
}
