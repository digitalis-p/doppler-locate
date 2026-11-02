// Native hot paths for dopploc (see dopploc_native.h). Builds two ways:
//   nvcc  -> CUDA kernels + OpenMP CPU fallback   (make)
//   g++ -x c++ -> OpenMP CPU only                 (make cpu)
#include "dopploc_native.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif

// --------------------------------------------------------------------------- shared math

HD static inline void interp_track(double tq, double t0, double dt, const double *P, const double *V,
                                   int n, double *p, double *v, int *inside) {
    double x = (tq - t0) / dt;
    int i;
    double w;
    if (x <= 0.0) {
        i = 0;
        w = 0.0;
    } else if (x >= n - 1) {
        i = n - 2;
        w = 1.0;
    } else {
        i = (int)x;
        w = x - i;
    }
    for (int k = 0; k < 3; k++) {
        p[k] = P[3 * i + k] + w * (P[3 * (i + 1) + k] - P[3 * i + k]);
        v[k] = V[3 * i + k] + w * (V[3 * (i + 1) + k] - V[3 * i + k]);
    }
    *inside = (x >= 0.0 && x <= n - 1);
}

// f/f0 heard at receive time tr (emission-time radial velocity). a = [c_e, w_s(3), c_p, w_m(3)]
// or NULL for still air at c.
HD static inline double ratio_at(double tr, const double *obs, double t0, double dt, const double *P,
                                 const double *V, int n, double c, const double *a, int *ok) {
    double ce = a ? a[0] : c, cp = a ? a[4] : c;
    double s0 = a ? a[1] : 0.0, s1 = a ? a[2] : 0.0, s2 = a ? a[3] : 0.0;
    double m0 = a ? a[5] : 0.0, m1 = a ? a[6] : 0.0, m2 = a ? a[7] : 0.0;
    double tau = tr - 2000.0 / cp, p[3], v[3];
    int inside;
    for (int it = 0; it < 60; it++) {
        interp_track(tau, t0, dt, P, V, n, p, v, &inside);
        double dx = p[0] - obs[0], dy = p[1] - obs[1], dz = p[2] - obs[2];
        double R = sqrt(dx * dx + dy * dy + dz * dz);
        double wm = (m0 * dx + m1 * dy + m2 * dz) / R;
        double tn = tr - R / (cp - wm);
        if (fabs(tn - tau) < 1e-5) {
            tau = tn;
            break;
        }
        tau = tn;
    }
    interp_track(tau, t0, dt, P, V, n, p, v, &inside);
    double dx = p[0] - obs[0], dy = p[1] - obs[1], dz = p[2] - obs[2];
    double R = sqrt(dx * dx + dy * dy + dz * dz);
    double vr = (dx * v[0] + dy * v[1] + dz * v[2]) / R;
    double ws = (s0 * dx + s1 * dy + s2 * dz) / R;
    *ok = inside;
    return (ce - ws) / (ce + vr - ws);
}

// CPA over samples [lo, hi) with parabolic refinement; returns 0 if not enough samples.
HD static inline int cpa_window(const double *obs, double t0, double dt, const double *P, int lo, int hi,
                                double c, double *tdel, double *dcpa) {
    if (hi - lo < 3) return 0;
    double best = 1e300;
    int k = lo;
    for (int i = lo; i < hi; i++) {
        double dx = P[3 * i] - obs[0], dy = P[3 * i + 1] - obs[1], dz = P[3 * i + 2] - obs[2];
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < best) {
            best = r2;
            k = i;
        }
    }
    double tau = t0 + k * dt, d2 = best;
    if (k > lo && k < hi - 1) {
        double r[3];
        for (int q = 0; q < 3; q++) {
            int i = k - 1 + q;
            double dx = P[3 * i] - obs[0], dy = P[3 * i + 1] - obs[1], dz = P[3 * i + 2] - obs[2];
            r[q] = dx * dx + dy * dy + dz * dz;
        }
        double den = r[0] - 2.0 * r[1] + r[2];
        if (den > 0) {
            double off = 0.5 * (r[0] - r[2]) / den;
            tau = t0 + (k + off) * dt;
            d2 = r[1] - 0.25 * (r[0] - r[2]) * off;
        }
    }
    double d = sqrt(d2 > 0 ? d2 : 0.0);
    *dcpa = d;
    *tdel = tau + d / c;
    return 1;
}

HD static inline void window_bounds(double t_expect, double win, double t0, double dt, int np, int *lo, int *hi) {
    int a = (int)((t_expect - win - t0) / dt);
    int b = (int)((t_expect + win - t0) / dt) + 1;
    *lo = a < 0 ? 0 : a;
    *hi = b > np ? np : b;
}

HD static inline float lerp_row(const float *Rrow, int nu, double u0, double du, double u) {
    double x = (u - u0) / du;
    if (!(x >= 0.0 && x <= nu - 1)) return NAN;
    int i = (int)x;
    if (i > nu - 2) i = nu - 2;
    double w = x - i;
    return (float)(Rrow[i] + w * (Rrow[i + 1] - Rrow[i]));
}

HD static inline void score_one_brightness(const float *Rrow, int nu, double u0, double du, double base,
                                           double delta, const double *f0s, int nf, const double *harms, int nh,
                                           const double *tau, int nt, const float *B, int nbin, double fb0,
                                           double dfb, float *S_out, int *A_out) {
    float bestv = -1.0f;
    int besti = 0;
    for (int f = 0; f < nf; f++) {
        double acc = 0.0;
        for (int j = 0; j < nt; j++) {
            float r = lerp_row(Rrow, nu, u0, du, tau[j] - base - delta);
            if (isnan(r)) continue;
            for (int h = 0; h < nh; h++) {
                double y = (harms[h] * f0s[f] * r - fb0) / dfb;
                if (y < 0.0 || y > nbin - 1) continue;
                int yi = (int)y;
                if (yi > nbin - 2) yi = nbin - 2;
                double w = y - yi;
                float b0 = B[(int64_t)yi * nt + j], b1 = B[(int64_t)(yi + 1) * nt + j];
                acc += b0 + w * (b1 - b0);
            }
        }
        float val = (float)(acc / ((double)nt * nh));
        if (val > bestv) {
            bestv = val;
            besti = f;
        }
    }
    *S_out = bestv;
    *A_out = besti;
}

HD static inline void score_one_trace(const float *Rrow, int nu, double u0, double du, double base, double delta,
                                      const double *f0s, int nf, const double *tt, const double *tf,
                                      const double *th, int nt, double sigma, float *S_out, int *A_out) {
    float bestv = -1.0f;
    int besti = 0;
    for (int f = 0; f < nf; f++) {
        double acc = 0.0;
        for (int j = 0; j < nt; j++) {
            float r = lerp_row(Rrow, nu, u0, du, tt[j] - base - delta);
            if (isnan(r)) continue;
            double e = (th[j] * f0s[f] * r - tf[j]) / sigma;
            acc += exp(-0.5 * e * e);
        }
        float val = (float)(acc / nt);
        if (val > bestv) {
            bestv = val;
            besti = f;
        }
    }
    *S_out = bestv;
    *A_out = besti;
}

// --------------------------------------------------------------------------- CPU (OpenMP)

static int model_cpu(const double *obs, const double *t_expect, const int32_t *fidx, int64_t n, const double *P,
                     const double *V, const int64_t *f_start, const int32_t *f_len, const double *f_t0,
                     const double *f_dt, double u0, double du, int nu, double c, double win,
                     const double *atm, float *R, double *t_cpa, double *d_cpa) {
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t i = 0; i < n; i++) {
        int fi = fidx[i];
        const double *Pf = P + 3 * f_start[fi], *Vf = V + 3 * f_start[fi];
        int np = f_len[fi];
        int lo, hi;
        window_bounds(t_expect[i], win, f_t0[fi], f_dt[fi], np, &lo, &hi);
        double tdel, d;
        float *Rrow = R + i * (int64_t)nu;
        if (!cpa_window(obs + 3 * i, f_t0[fi], f_dt[fi], Pf, lo, hi, c, &tdel, &d)) {
            t_cpa[i] = NAN;
            d_cpa[i] = NAN;
            for (int j = 0; j < nu; j++) Rrow[j] = NAN;
            continue;
        }
        t_cpa[i] = tdel;
        d_cpa[i] = d;
        for (int j = 0; j < nu; j++) {
            int ok;
            double r = ratio_at(tdel + u0 + j * du, obs + 3 * i, f_t0[fi], f_dt[fi], Pf, Vf, np, c,
                                atm ? atm + 8 * i : 0, &ok);
            Rrow[j] = ok ? (float)r : NAN;
        }
    }
    return 0;
}

static int brightness_cpu(const float *R, int64_t n, int nu, double u0, double du, const double *base,
                          const double *deltas, int nd, const double *f0s, int nf, const double *harms, int nh,
                          const double *tau, int nt, const float *B, int nbin, double fb0, double dfb, float *S,
                          int32_t *A) {
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (int64_t i = 0; i < n; i++)
        for (int d = 0; d < nd; d++) {
            int a;
            score_one_brightness(R + i * (int64_t)nu, nu, u0, du, base[i], deltas[d], f0s, nf, harms, nh, tau, nt,
                                 B, nbin, fb0, dfb, &S[i * nd + d], &a);
            A[i * nd + d] = a;
        }
    return 0;
}

static int trace_cpu(const float *R, int64_t n, int nu, double u0, double du, const double *base,
                     const double *deltas, int nd, const double *f0s, int nf, const double *tt, const double *tf,
                     const double *th, int nt, double sigma, float *S, int32_t *A) {
#pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (int64_t i = 0; i < n; i++)
        for (int d = 0; d < nd; d++) {
            int a;
            score_one_trace(R + i * (int64_t)nu, nu, u0, du, base[i], deltas[d], f0s, nf, tt, tf, th, nt, sigma,
                            &S[i * nd + d], &a);
            A[i * nd + d] = a;
        }
    return 0;
}

// --------------------------------------------------------------------------- CUDA

#ifdef __CUDACC__

#define CK(x)                                   \
    do {                                        \
        cudaError_t e_ = (x);                   \
        if (e_ != cudaSuccess) return (int)e_;  \
    } while (0)

template <typename T>
static int upload(T **d, const T *h, size_t count) {
    CK(cudaMalloc((void **)d, count * sizeof(T) + 8));
    if (h) CK(cudaMemcpy(*d, h, count * sizeof(T), cudaMemcpyHostToDevice));
    return 0;
}

__global__ void k_cpa(const double *obs, const double *t_expect, const int32_t *fidx, int64_t n, const double *P,
                      const int64_t *f_start, const int32_t *f_len, const double *f_t0, const double *f_dt,
                      double c, double win, double *t_cpa, double *d_cpa) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int fi = fidx[i];
    int lo, hi;
    window_bounds(t_expect[i], win, f_t0[fi], f_dt[fi], f_len[fi], &lo, &hi);
    double tdel, d;
    if (cpa_window(obs + 3 * i, f_t0[fi], f_dt[fi], P + 3 * f_start[fi], lo, hi, c, &tdel, &d)) {
        t_cpa[i] = tdel;
        d_cpa[i] = d;
    } else {
        t_cpa[i] = NAN;
        d_cpa[i] = NAN;
    }
}

__global__ void k_ratio(const double *obs, const int32_t *fidx, int64_t n, const double *P, const double *V,
                        const int64_t *f_start, const int32_t *f_len, const double *f_t0, const double *f_dt,
                        double u0, double du, int nu, double c, const double *atm, const double *t_cpa,
                        float *R) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n * (int64_t)nu) return;
    int64_t i = idx / nu;
    int j = (int)(idx % nu);
    double tdel = t_cpa[i];
    if (isnan(tdel)) {
        R[idx] = NAN;
        return;
    }
    int fi = fidx[i];
    int ok;
    double r = ratio_at(tdel + u0 + j * du, obs + 3 * i, f_t0[fi], f_dt[fi], P + 3 * f_start[fi],
                        V + 3 * f_start[fi], f_len[fi], c, atm ? atm + 8 * i : 0, &ok);
    R[idx] = ok ? (float)r : NAN;
}

__global__ void k_brightness(const float *R, int64_t n, int nu, double u0, double du, const double *base,
                             const double *deltas, int nd, const double *f0s, int nf, const double *harms, int nh,
                             const double *tau, int nt, const float *__restrict__ B, int nbin, double fb0,
                             double dfb, float *S, int32_t *A) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n * (int64_t)nd) return;
    int64_t i = idx / nd;
    int d = (int)(idx % nd);
    int a;
    score_one_brightness(R + i * (int64_t)nu, nu, u0, du, base[i], deltas[d], f0s, nf, harms, nh, tau, nt, B, nbin,
                         fb0, dfb, &S[idx], &a);
    A[idx] = a;
}

__global__ void k_trace(const float *R, int64_t n, int nu, double u0, double du, const double *base,
                        const double *deltas, int nd, const double *f0s, int nf, const double *tt, const double *tf,
                        const double *th, int nt, double sigma, float *S, int32_t *A) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n * (int64_t)nd) return;
    int64_t i = idx / nd;
    int d = (int)(idx % nd);
    int a;
    score_one_trace(R + i * (int64_t)nu, nu, u0, du, base[i], deltas[d], f0s, nf, tt, tf, th, nt, sigma, &S[idx],
                    &a);
    A[idx] = a;
}

static int grid(int64_t work, int block) { return (int)((work + block - 1) / block); }

static int model_gpu(const double *obs, const double *t_expect, const int32_t *fidx, int64_t n, const double *P,
                     const double *V, const int64_t *f_start, const int32_t *f_len, const double *f_t0,
                     const double *f_dt, int nflights, int64_t nrows, double u0, double du, int nu, double c,
                     double win, const double *atm, float *R, double *t_cpa, double *d_cpa, int device) {
    CK(cudaSetDevice(device));
    double *dP = 0, *dV = 0, *dt0 = 0, *ddt = 0;
    int64_t *dstart = 0;
    int32_t *dlen = 0;
    if (upload(&dP, P, 3 * nrows) || upload(&dV, V, 3 * nrows) || upload(&dstart, f_start, nflights) ||
        upload(&dlen, f_len, nflights) || upload(&dt0, f_t0, nflights) || upload(&ddt, f_dt, nflights))
        return -1;
    const int64_t chunk = 1 << 16;
    double *dobs = 0, *dte = 0, *dtc = 0, *ddc = 0, *dw = 0;
    int32_t *dfi = 0;
    float *dR = 0;
    int64_t cmax = n < chunk ? n : chunk;
    if (upload(&dobs, (double *)0, 3 * cmax) || upload(&dte, (double *)0, cmax) || upload(&dfi, (int32_t *)0, cmax) ||
        upload(&dtc, (double *)0, cmax) || upload(&ddc, (double *)0, cmax) || upload(&dR, (float *)0, cmax * nu) ||
        (atm && upload(&dw, (double *)0, 8 * cmax)))
        return -1;
    for (int64_t s = 0; s < n; s += chunk) {
        int64_t m = n - s < chunk ? n - s : chunk;
        CK(cudaMemcpy(dobs, obs + 3 * s, 3 * m * sizeof(double), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dte, t_expect + s, m * sizeof(double), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dfi, fidx + s, m * sizeof(int32_t), cudaMemcpyHostToDevice));
        if (atm) CK(cudaMemcpy(dw, atm + 8 * s, 8 * m * sizeof(double), cudaMemcpyHostToDevice));
        k_cpa<<<grid(m, 128), 128>>>(dobs, dte, dfi, m, dP, dstart, dlen, dt0, ddt, c, win, dtc, ddc);
        k_ratio<<<grid(m * nu, 256), 256>>>(dobs, dfi, m, dP, dV, dstart, dlen, dt0, ddt, u0, du, nu, c, dw, dtc, dR);
        CK(cudaGetLastError());
        CK(cudaMemcpy(t_cpa + s, dtc, m * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(d_cpa + s, ddc, m * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(R + s * nu, dR, m * nu * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaFree(dP); cudaFree(dV); cudaFree(dstart); cudaFree(dlen); cudaFree(dt0); cudaFree(ddt);
    cudaFree(dobs); cudaFree(dte); cudaFree(dfi); cudaFree(dtc); cudaFree(ddc); cudaFree(dR); cudaFree(dw);
    return 0;
}

static int sweep_gpu(int trace, const float *R, int64_t n, int nu, double u0, double du, const double *base,
                     const double *deltas, int nd, const double *f0s, int nf, const double *harms, int nh,
                     const double *tau, int nt, const float *B, int nbin, double fb0, double dfb, const double *tt,
                     const double *tf, const double *th, double sigma, float *S, int32_t *A, int device) {
    CK(cudaSetDevice(device));
    double *ddel = 0, *df0 = 0, *dh = 0, *dtau = 0, *dtt = 0, *dtf = 0, *dth = 0;
    float *dB = 0;
    if (upload(&ddel, deltas, nd) || upload(&df0, f0s, nf)) return -1;
    if (trace) {
        if (upload(&dtt, tt, nt) || upload(&dtf, tf, nt) || upload(&dth, th, nt)) return -1;
    } else {
        if (upload(&dh, harms, nh) || upload(&dtau, tau, nt) || upload(&dB, B, (size_t)nbin * nt)) return -1;
    }
    const int64_t chunk = 1 << 16;
    int64_t cmax = n < chunk ? n : chunk;
    float *dR = 0, *dS = 0;
    double *dbase = 0;
    int32_t *dA = 0;
    if (upload(&dR, (float *)0, cmax * nu) || upload(&dbase, (double *)0, cmax) || upload(&dS, (float *)0, cmax * nd) ||
        upload(&dA, (int32_t *)0, cmax * nd))
        return -1;
    for (int64_t s = 0; s < n; s += chunk) {
        int64_t m = n - s < chunk ? n - s : chunk;
        CK(cudaMemcpy(dR, R + s * nu, m * nu * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dbase, base + s, m * sizeof(double), cudaMemcpyHostToDevice));
        if (trace)
            k_trace<<<grid(m * nd, 128), 128>>>(dR, m, nu, u0, du, dbase, ddel, nd, df0, nf, dtt, dtf, dth, nt, sigma,
                                                dS, dA);
        else
            k_brightness<<<grid(m * nd, 128), 128>>>(dR, m, nu, u0, du, dbase, ddel, nd, df0, nf, dh, nh, dtau, nt, dB,
                                                     nbin, fb0, dfb, dS, dA);
        CK(cudaGetLastError());
        CK(cudaMemcpy(S + s * nd, dS, m * nd * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(A + s * nd, dA, m * nd * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
    cudaFree(ddel); cudaFree(df0); cudaFree(dh); cudaFree(dtau); cudaFree(dtt); cudaFree(dtf); cudaFree(dth);
    cudaFree(dB); cudaFree(dR); cudaFree(dS); cudaFree(dbase); cudaFree(dA);
    return 0;
}
#endif

// --------------------------------------------------------------------------- API

extern "C" int dl_cuda_device_count(void) {
#ifdef __CUDACC__
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
#else
    return 0;
#endif
}

extern "C" int dl_model(const double *obs, const double *t_expect, const int32_t *fidx, int64_t n, const double *P,
                        const double *V, const int64_t *f_start, const int32_t *f_len, const double *f_t0,
                        const double *f_dt, int32_t nflights, int64_t nrows, double u0, double du, int32_t nu,
                        double c, double win, const double *atm, float *R, double *t_cpa, double *d_cpa,
                        int device) {
#ifdef __CUDACC__
    if (device >= 0)
        return model_gpu(obs, t_expect, fidx, n, P, V, f_start, f_len, f_t0, f_dt, nflights, nrows, u0, du, nu, c, win,
                         atm, R, t_cpa, d_cpa, device);
#endif
    (void)nflights;
    (void)nrows;
    return model_cpu(obs, t_expect, fidx, n, P, V, f_start, f_len, f_t0, f_dt, u0, du, nu, c, win, atm, R, t_cpa,
                     d_cpa);
}

extern "C" int dl_score_brightness(const float *R, int64_t n, int32_t nu, double u0, double du, const double *base,
                                   const double *deltas, int32_t nd, const double *f0s, int32_t nf,
                                   const double *harms, int32_t nh, const double *tau, int32_t nt, const float *B,
                                   int32_t nbin, double fb0, double dfb, float *S, int32_t *A, int device) {
#ifdef __CUDACC__
    if (device >= 0)
        return sweep_gpu(0, R, n, nu, u0, du, base, deltas, nd, f0s, nf, harms, nh, tau, nt, B, nbin, fb0, dfb, 0, 0,
                         0, 0.0, S, A, device);
#endif
    return brightness_cpu(R, n, nu, u0, du, base, deltas, nd, f0s, nf, harms, nh, tau, nt, B, nbin, fb0, dfb, S, A);
}

extern "C" int dl_score_trace(const float *R, int64_t n, int32_t nu, double u0, double du, const double *base,
                              const double *deltas, int32_t nd, const double *f0s, int32_t nf, const double *tt,
                              const double *tf, const double *th, int32_t nt, double sigma, float *S, int32_t *A,
                              int device) {
#ifdef __CUDACC__
    if (device >= 0)
        return sweep_gpu(1, R, n, nu, u0, du, base, deltas, nd, f0s, nf, 0, 0, 0, nt, 0, 0, 0.0, 1.0, tt, tf, th,
                         sigma, S, A, device);
#endif
    return trace_cpu(R, n, nu, u0, du, base, deltas, nd, f0s, nf, tt, tf, th, nt, sigma, S, A);
}
