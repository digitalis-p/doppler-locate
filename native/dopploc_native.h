/* Native hot paths for dopploc: forward Doppler model and spectrogram scoring sweeps.
 *
 * Conventions match src/dopploc/physics.py and score.py:
 *   tracks are uniformly sampled (t0 + k*dt) ECEF positions P and air-relative velocities V,
 *   concatenated for many flights (row offset f_start[i], length f_len[i]);
 *   f/f0 = c / (c + v_r) with v_r evaluated at the emission time t_e solving
 *   t_r = t_e + |P(t_e) - obs| / c.
 * Every function returns 0 on success. device < 0 selects the CPU (OpenMP) implementation.
 */
#ifndef DOPPLOC_NATIVE_H
#define DOPPLOC_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int dl_cuda_device_count(void);

/* For each candidate i: CPA of flight fidx[i] within +-win s of t_expect[i] (delayed CPA time
 * t_cpa[i], slant distance d_cpa[i]) and the ratio f/f0 heard at receive times
 * t_cpa[i] + u0 + j*du, j < nu, written to R[i*nu + j] (NaN outside the track).
 * atm: per-candidate atmosphere rows (n x 8) [c_e, w_s(3), c_p, w_m(3)] or NULL (still air at c);
 * with d = unit(obs -> aircraft): f/f0 = (c_e - w_s.d) / (c_e + v.d - w_s.d), travel time
 * R / (c_p - w_m.d). */
int dl_model(const double *obs, const double *t_expect, const int32_t *fidx, int64_t n,
             const double *P, const double *V, const int64_t *f_start, const int32_t *f_len,
             const double *f_t0, const double *f_dt, int32_t nflights, int64_t nrows,
             double u0, double du, int32_t nu, double c, double win, const double *atm,
             float *R, double *t_cpa, double *d_cpa, int device);

/* Brightness score. For candidate i, shift deltas[d] and fundamental f0s[f]:
 *   mean over frames j and harmonics h of B(h*f0*r_ij, tau_j)
 *   with r_ij = R_i(tau_j - base[i] - deltas[d]) (linear interpolation on the u grid).
 * Writes the best mean over f0s to S[i*nd + d] and its index to A[i*nd + d]. */
int dl_score_brightness(const float *R, int64_t n, int32_t nu, double u0, double du,
                        const double *base, const double *deltas, int32_t nd,
                        const double *f0s, int32_t nf, const double *harms, int32_t nh,
                        const double *tau, int32_t nt, const float *B, int32_t nbin,
                        double fb0, double dfb, float *S, int32_t *A, int device);

/* Trace score: mean over traced points of exp(-0.5 ((h_j f0 r - f_j)/sigma)^2). */
int dl_score_trace(const float *R, int64_t n, int32_t nu, double u0, double du,
                   const double *base, const double *deltas, int32_t nd,
                   const double *f0s, int32_t nf, const double *tt, const double *tf,
                   const double *th, int32_t nt, double sigma, float *S, int32_t *A, int device);

#ifdef __cplusplus
}
#endif
#endif
