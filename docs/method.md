# Method

This page derives what `ExtrinsicCalibrator` does and records the design decisions behind
it. Conventions follow Aether6: Hamilton quaternions stored `[w, x, y, z]`, `q_AB` maps
B-frame vectors into frame A, world frame NED with gravity `g^W = [0, 0, +g]`, IMU frame
forward-right-down, camera frame x-right / y-down / z-optical-axis. Every rotation error is a
**right** perturbation, `R = R̂ Exp(δθ)`.

## 1. Problem

Recover the rigid transform `T_IC = (R_IC, p_IC)` of the left camera in the IMU frame (and,
optionally, the camera-IMU clock offset `t_d`) from a stereo/IMU stream alone. The intrinsics
and stereo baseline are known; the landmarks are **not** (no target), and nothing about the
truth trajectory or the truth extrinsic reaches the estimator. It starts from a nominal
("CAD") mount that is wrong by 5° and 5 cm. In the evaluation this starting value is made by
perturbing each run's hidden truth by exactly 5° and 5 cm in a random direction, so that the
initial error is controlled precisely. The estimator only ever receives the perturbed value.

## 2. State

The minimum needed to *use* the IMU between frames is its pose, velocity and biases; the
minimum needed to *form a reprojection residual* without landmark states is one earlier pose
to triangulate against. So the state is the IMU navigation state, the calibration
parameters, and a single stochastic clone:

| block | nominal | error (dim) |
|---|---|---|
| IMU position, velocity | `p, v` (world) | `δp, δv` (3+3) |
| IMU attitude | `q_WI` | `δθ` (3), body frame |
| gyro / accel bias | `b_g, b_a` | `δb_g, δb_a` (3+3) |
| **extrinsic rotation** | `q_IC` | `δφ` (3), camera frame |
| **extrinsic lever arm** | `p_IC` | `δp_IC` (3) |
| **time offset** | `t_d` | `δt_d` (1) |
| clone of the IMU pose at the previous frame | `q_c, p_c` | `δθ_c, δp_c` (3+3) |

28 error states. There are no landmark states and no sliding window, and the first five
blocks keep Aether6's ordering so its error dynamics carry over unchanged.

## 3. Propagation

Between IMU samples `a → b` the nominal state is integrated with the trapezoidal rule
(midpoint rate for the attitude, average of the rotated specific forces for velocity and
position), so a 200 Hz stream integrates with O(dt³) local error. The error dynamics are
Aether6's:

```
δṗ = δv
δv̇ = −R [f]× δθ − R δb_a − R n_a
δθ̇ = −[ω]× δθ − δb_g − n_g
δḃ_g = n_bg,   δḃ_a = n_ba
δφ̇ = n_φ,      δṗ_IC = n_p,     δṫ_d = n_t      (random walks)
δθ̇_c = 0,      δṗ_c = 0                         (the clone is a snapshot)
```

discretised with Aether6's second-order series (`estimation::discretize`). Only the 15-state
navigation block has dynamics, so the covariance is propagated block-wise:
`P_nn ← Φ P_nn Φᵀ + Q_d`, `P_nr ← Φ P_nr`, and the rest only receives its random-walk noise.

The IMU white noise is configured as Aether6's sensor model specifies it, a per-sample σ at
the IMU rate. The equivalent continuous spectral density is `σ² / rate`. (Aether6's own
navigation EKF feeds the per-sample σ straight into `Q_c`, which over-states that term by a
factor of `rate`, i.e. 200×. That is harmless, and merely conservative, for its GNSS-aided
filter; the calibrator converts it.)

**Extrinsic random walk.** `n_φ` and `n_p` are what let the filter keep tracking a mount that
keeps moving. With zero intensity the covariance collapses as data accumulates and the
estimate freezes: the "calibrate once" behaviour. The intensity sets a lag/noise trade-off
(§7).

## 4. Measurement: stereo reprojection with the landmark marginalised

At each stereo frame `k` (exposure at IMU time `τ_k = stamp_k + t̂_d`; the IMU is
interpolated to that instant), every landmark seen in both frame `k−1` (the clone) and
frame `k` gives 8 pixel measurements: left and right at both frames. For a view with IMU pose
`(R, p)`:

```
q = Rᵀ (f − p)               point in the IMU frame
P = R_ICᵀ (q − p_IC)          point in the left camera
h = [π(P); π(P − b e_x)]      left and right pixels
```

with `J = ∂h/∂P` (4×3) and `A = R̂_ICᵀ`, the Jacobians are (`MeasurementModel.hpp`, each
verified against central differences in `tests/test_measurement_model.cpp`):

| w.r.t. | Jacobian |
|---|---|
| `δθ` (view attitude) | `J A [q̂]×` |
| `δp` (view position) | `−J A R̂ᵀ` |
| `δφ` (extrinsic rotation) | `J [P̂]×` |
| `δp_IC` (lever arm) | `−J A` |
| `δf` (landmark) | `J A R̂ᵀ` |
| `δt_d` | `J A ([q̂]× ω̂ − R̂ᵀ v̂)` |

The last row is the exposure-time model: the image was taken at `τ + δt_d`, when the pose
was `(R̂ Exp(ω̂ δt_d), p̂ + v̂ δt_d)`.

**Landmark.** `f` is linearised at a point triangulated from the clone's stereo pair and
refined by three Gauss-Newton steps over all four views. It is then **marginalised exactly**:
with `H_f` the 8×3 landmark Jacobian, the residual and state Jacobian are projected onto
the 5-dimensional left null space `N` of `H_f` (the last five columns of its Householder Q).
`N` is orthonormal, so isotropic pixel noise stays isotropic. This is the MSCKF trick
applied to a two-pose window, and it is why the landmarks never need to enter the state.

**Timing.** The time offset is estimated in every scenario. The lever arm is so tightly
coupled to timing that an unmodelled 2.5 ms camera lag biases it by 1.5 cm and breaks
convergence (`time_offset_ignored`). Two details keep the timing exact:
* When an update moves `t̂_d` by `Δ`, the filter time (the exposure as estimated *before* the
  update) is off by `Δ`. The clone is therefore re-timed to `stamp + t̂_d` (pose shifted by
  `(ω̂Δ, v̂Δ)`) before it becomes the next anchor.
* Frames may be delivered after their exposure, as from a real driver. The filter keeps a
  short IMU buffer and runs `max_frame_latency` behind the newest sample, so a late frame is
  still fused at its exposure time. The evaluation delivers every frame 30 ms late with an
  80 ms lag, and the results are bit-identical to on-time delivery (`tests/test_calibrator.cpp`).

**Robustness.** Each landmark's 5-dof residual is gated at χ²(5, 99.9 %) = 20.5 against
`S = H P Hᵀ + σ²I`, which removes wrong correspondences.

**Update.** The ~100 surviving landmarks give ~500 rows. They are QR-compressed to at most 28
(`H = Q [T; 0]`, so only `T` and `Qᵀr` matter), and Aether6's Joseph-form update is applied
(`estimation::josephUpdate`). Then come the multiplicative reset of the three rotation blocks
(`G = I − ½[δθ]×` each) and moving the clone to the current pose: `P ← J P Jᵀ`, with `J`
copying the current-pose rows into the clone slots.

## 5. Initialisation (no ground truth)

The rig is held still for the first 2 s. The filter averages 1.5 s of IMU data and checks
stillness (gyro σ < 0.02 rad/s, accel σ < 0.2 m/s²). It then sets `b_g` to the mean rate and
roll/pitch from the mean specific force. Yaw and position are set to zero, which defines the
estimator's world frame (both are unobservable and irrelevant to the extrinsic). `b_a = 0`,
and the extrinsic starts at the nominal mount.

## 6. Observability

The extrinsic enters the residual only through how the camera's motion relates to the IMU's.
The rotation is observable as soon as the rig rotates about two non-parallel axes: hand-eye
`ΔR_C = R_IC^T ΔR_I R_IC`, plus the direction of translation. The lever arm enters through the
rotation-induced camera velocity `ω × p_IC`. A rotation about a single fixed axis `a` never
moves a point on that axis, so **the lever-arm component along `a` is unobservable under
single-axis rotation**. That is why the figure-eight carries roll and pitch excitation as
well as yaw. The `yaw_only` scenario removes roll and pitch and shows exactly this: the filter
keeps a large σ along the yaw axis and the lever-arm error there stays near its prior.

## 7. Tracking drift and recovering from knocks

**Slow drift.** A random-walk model with intensity `q` and measurement information rate `J`
has a steady-state time constant `τ ≈ 1/√(qJ)` and noise `σ² ≈ √(q/J)`. A mount drifting at a
constant rate `ṙ` is followed with a lag of about `ṙ τ`. The rotation intensity was picked by
sweeping it on the drift scenario with the final configuration (20 runs per value):

| `extrinsic_rotation_walk` [rad/√s] | 1e-5 | 2e-5 | 5e-5 | **1e-4** | 2e-4 |
|---|---|---|---|---|---|
| max tracking error p95 [deg] | 0.823 | 0.116 | 0.063 | **0.053** | 0.064 |
| RMS tracking error p50 [deg] | 0.122 | 0.091 | 0.040 | **0.025** | 0.024 |
| mean NEES (6 = consistent) | 373 | 97 | 8.9 | **2.9** | 2.0 |
| rotation error at 60 s p95 [deg] | 0.016 | 0.016 | 0.017 | **0.023** | 0.033 |
| runs tracked within 0.3° | 1 / 20 | 20 / 20 | 20 / 20 | **20 / 20** | 20 / 20 |

Below 1e-4 the filter lags the drift and becomes over-confident (NEES ≫ 6). At 1e-5 the drift
is so far outside the model that the change detector takes it for a knock and re-opens the
covariance in every run, and the error spikes while the filter re-converges. Above 1e-4 noise
dominates. At 1e-4 the drift is fully explained by the model: NEES matches the stationary
case, and the converged accuracy on a stationary mount barely changes.

**Knocks.** A sudden step is far outside what the random walk explains, but in a two-frame
residual it barely shows in the NIS: an extrinsic error only reveals itself through the
rig's motion over 50 ms. The detector therefore looks at the extrinsic directly. Over the last
`W` updates (40 = 2 s), the summed extrinsic corrections `Δ = Σ K_i ν_i` of a consistent
filter with white innovations would have covariance `C = Σ (P⁻_i − P⁺_i)`, and
`T = Δᵀ C⁻¹ Δ` would be χ²₆ while the mount behaves as modelled. Here the innovations of
consecutive pairs are slightly (negatively) correlated and the filter is conservative (§8),
so `T` is a well-scaled heuristic rather than an exact χ²₆ test, and its threshold is
validated empirically (no false alarms, below). After a step the filter is dragged
in one direction update after update and `T` grows quadratically. When `T > 40` (a χ²₆ tail
probability of 4.6·10⁻⁷), the extrinsic covariance is re-opened (correlations dropped,
σ = 1° / 1 cm) and the filter re-converges to the new mount within seconds.

Two details keep it quiet when nothing has happened:

* It is **armed only once the calibration has converged** (every rotation σ < 0.1° and lever-arm
  σ < 5 mm). Neither the initial convergence nor the re-convergence after a reset can then be
  mistaken for a knock. Before this rule, 3 / 50 time-offset runs fired spuriously while
  `t_d` was still settling.
* The re-open size is matched to a knock, not to the CAD prior. Re-opening to 5° / 5 cm made
  the first updates overshoot: up to 3.8° of rotation error after a 1.5° knock. With
  1° / 1 cm the rotation error never exceeds the knock itself. The lever arm still overshoots
  for a few seconds (3.0 cm p50 for a 1.95 cm knock) while the rotation settles. A 5° / 5 cm
  knock recovers in 5.3 s (p50).

Because the filter is conservative, `T` sits below its nominal χ²₆ distribution on an
undisturbed mount. There were no false alarms in the 200 runs where it was armed and the
model was right (README). When the model is *wrong* it fires repeatedly: in
`time_offset_ignored` (an unmodelled 2.5 ms camera lag) it fires ~6 times per run. That is
the behaviour a self-checking calibrator should have.

## 8. Consistency, honestly

* **Vision NIS per dof ≈ 1.00.** The pixel noise model matches the data.
* **Extrinsic NEES ≈ 2.0** against 6 expected on a stationary mount (2.7 with the random walk
  switched off), so the filter is **conservative** by about 1.4–1.7× in σ. Every frame is used
  twice, as the "current" view of one update and the anchor of the next, so consecutive
  residuals share pixel noise with *negative* correlation. A filter that assumes independent
  residuals under-counts the information for slowly varying parameters such as the
  extrinsic. Fusing disjoint frame pairs only (`--set disjoint_pairs=1`, every observation used
  once) brings NEES to 5.4, but discards half the data and doubles the error (p95 0.051° /
  0.65 cm instead of 0.023° / 0.32 cm). The default keeps the accuracy and errs on the safe side.
* **Noise-free data with a perfect prior stays at truth** (1e-4° / 0.09 mm after 60 s). This
  is the regression test for model mismatches between the simulator and the estimator. During
  development it exposed a real one: `propagate()` took its first IMU sample by const
  reference to a member it then overwrote, so every step integrated the *end* sample twice.
  That is a hidden 2.5 ms IMU-camera time shift, worth ~2 mrad of attitude error at
  1 rad/s, and it showed up as a systematic, seed-independent 0.3° extrinsic error during
  convergence. `tests/test_calibrator.cpp` now pins propagation accuracy and its O(dt²)
  convergence order.
