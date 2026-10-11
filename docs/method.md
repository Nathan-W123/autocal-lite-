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
coupled to timing that an unmodelled 2.5 ms camera lag biases it by about 1 cm (p50; 3.5 cm
p95) and breaks convergence in more than half of the runs (`time_offset_ignored`). Two details keep the timing exact:
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

**Pixel noise, estimated online.** Write the stacked residual of the `n` matched landmarks as
`r = H δx + n`, with `n ~ N(0, λσ₀²I)`, where `σ₀` is the configured pixel noise and `λ` the
unknown ratio to the truth. With `H = QR` (Householder, 28 columns), projecting `r` off the
first 28 columns of `Q` removes every state direction, whatever the rank of `H`. What is left,
`e`, holds `5n − 28` degrees of freedom of pure noise: `E‖e‖² = (5n − 28) λσ₀²`, whatever the
state error and whatever `P` says. Landmark `j`'s share of `e` has energy `≈ λσ₀² χ²_k` with
`k = (5n − 28)/n ≈ 4.7`. The median over landmarks, divided by the median of `χ²_k`
(Wilson–Hilferty: `k (1 − 2/9k)³`), therefore estimates `λσ₀²`. The first frame sets the
estimate and a running mean over 20 frames follows it. `R = λ̂σ₀² I` is then used for both the
gate and the update. Three choices matter, and each was found by breaking an earlier version:

* **Before the gate.** The first version measured the noise after the per-landmark gate, from
  the residual the update leaves, and started from the configured value. With the noise
  configured 4× or more below the truth, the gate (sized by the wrong noise) rejected most
  landmarks. The few updates that passed collapsed `P`, and by the time the estimate had caught
  up the covariance was over-confident for good: the status then certified errors outside
  tolerance. Measured from the data before the gate, the first frame already has it right.
* **Robust to wrong correspondences.** A wrong match is hundreds of pixels. Through the
  projection it leaks into every landmark's residual, so even the median inflates (to σ ≈ 5 px
  with 5 % mismatches; the outlier regression test caught this). Gross mismatches are therefore
  screened out first, by the landmark gate evaluated at 10× the current noise σ. That is wide
  enough that a noise value wrong by up to the 10× clamp cannot reject clean landmarks. Then
  any landmark whose energy is beyond χ²₅(99.9 %) of the estimated level is dropped and the
  projection repeated (at most 3 passes).
* **Independent of `P`.** The estimate never uses the covariance, so a wrong covariance cannot
  talk it into a wrong noise level.

Results, from the committed 50-run campaigns:

* **Configured right** (0.5 px; the `convergence`, `drift`, `time_offset`, `outliers`,
  `yaw_only` and `knock` campaigns): the final estimate is 0.490–0.516 px, with
  p50 0.503. The median-based estimate runs about 0.5 % high.
* **Configured at half the truth** (`noise_misspecified`: 1.0 px real, 0.5 px configured): the
  estimate is 0.990–1.023 px, NEES is 1.4, and all 50 runs converge. With the estimation off
  (`noise_misspecified_fixed`), the same runs are confidently wrong: NEES 23, and only 33 / 50
  runs are within spec.
* **Configured 10× too low or too high** (0.05 or 5 px), with or without 5 % wrong
  correspondences: every run converges with unchanged accuracy, and the status never
  certifies a wrong calibration (README sensitivity checks). The estimates there lie on one
  side of 0.500 px, because the truth sits exactly on the 10× clamp.

`tests/test_status.cpp` checks single runs at 1×, 2×, ½× and 7× either way, with and without
wrong correspondences.

Only the pixel noise is estimated. The IMU noise parameters are not: they enter through `Q`,
whose errors are not separable from the state error in the same clean way.

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
sweeping it on the drift scenario (20 runs per value). The sweep was run during development,
before the online noise estimate; it can be repeated with
`autocal_mc --scenario drift --trials 20 --set rot_walk=<value>`:

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
probability of 4.6·10⁻⁷), the covariance is re-opened and the filter re-converges to the new
mount within seconds.

It is **armed only once the calibration has converged** (every rotation σ < 0.1° and lever-arm
σ < 5 mm), so neither the initial convergence nor the re-convergence after a reset can be
mistaken for a knock. Before this rule, 3 / 50 time-offset runs fired spuriously while `t_d`
was still settling.

Two details make the re-open right, and the status's self-assessment (§9) is what exposed both:

* **Its size is matched to a knock**, not to the CAD prior: the extrinsic is re-opened to
  σ = 2° / 2 cm, with its correlations dropped. Re-opening to 5° / 5 cm made the first updates
  overshoot: up to 3.8° of rotation error after a 1.5° knock. 1° / 1 cm (the first default)
  was too tight for a 5° / 5 cm knock: the filter re-converged over-confident, and the status
  re-certified a lever arm still more than 1 cm off. Over 50 runs of a 5° / 5 cm knock, 1° / 1 cm gave
  283 false-claim frames after the detection, and 2° / 2 cm gives none. It also recovers
  faster: back in tolerance after 3.6 s instead of 8.2 s (p50), and 2.6 s for the 1.5° knock
  (3.0 s with 1° / 1 cm). The price is some overshoot. After the 1.5° / 1.95 cm knock, the
  rotation error peaks above the knock in 13 / 50 runs (at most 2.0°), and the lever arm
  overshoots to 2.9 cm (p50; 2.2 cm with 1° / 1 cm) for a few seconds. Both happen while the
  status says *recalibrating*.
* **The navigation states are loosened too.** Between the knock and its detection (0.2–0.4 s),
  every update pushes part of the knock into the attitude, velocity and biases, with full
  confidence. Re-opening only the extrinsic leaves that error in states the filter believes it
  knows, so it re-converges over-confident. A re-open therefore also adds 0.5° of attitude,
  2 cm/s of velocity, 1e-3 rad/s of gyro bias and 0.05 m/s² of accel bias σ, in quadrature.
  Without it, the
  status re-certified a lever arm still more than 1 cm off, several seconds after the
  detection: 135 frames over 50 runs of the 1.5° knock, and 320 of the 5° knock. With it there
  are none. The only false claims left are the frames between a knock and its detection
  (0.25 s, p50).

Because the filter is conservative, `T` sits below its nominal χ²₆ distribution on an
undisturbed mount. There were no false alarms in the 200 independent runs where it was armed
and the model was right (`drift`, `time_offset`, `outliers`, `noise_misspecified`; README),
nor before the knock in any of the 50 knock runs. When the model is *wrong* it fires repeatedly: in
`time_offset_ignored` (an unmodelled 2.5 ms camera lag) it fires 3 times per run. That is
the behaviour a self-checking calibrator should have.

## 8. Consistency, honestly

* **Vision NIS, in two parts.** Since the noise is estimated (§4), the innovation test is
  split. The residual outside the range of `H`, over the noise in use, tests the noise level
  alone; it is ≈ 1 almost by construction (0.983–0.987 in every campaign where it is estimated), so it only confirms the
  estimator. The ≤ 28 compressed rows test `H P Hᵀ + R`, and their NIS is 0.986–0.991 per
  dof. Once converged `H P Hᵀ` is small next to `R`, so that is a weak test of `P`: `knock`
  (NEES 11) still shows 0.989, and `drift_frozen` (NEES 13 000) shows 1.03. The real
  test of `P` is the NEES, which needs the truth and so exists only in simulation.
* **Extrinsic NEES ≈ 1.9** against 6 expected on a stationary mount (2.5 with the random walk
  switched off), so the filter is **conservative** by about 1.5–1.8× in σ. Every frame is used
  twice, as the "current" view of one update and the anchor of the next, so consecutive
  residuals share pixel noise with *negative* correlation. A filter that assumes independent
  residuals under-counts the information for slowly varying parameters such as the
  extrinsic. Fusing disjoint frame pairs only (`--set disjoint_pairs=1`, every observation used
  once) brings NEES to 5.3, but discards half the data and doubles the error (p95 0.045° /
  0.73 cm instead of 0.025° / 0.34 cm). The default keeps the accuracy and errs on the safe side.
  (Both variants are in the README sensitivity checks.)
* **Noise-free data with a perfect prior stays at truth** (1e-4° / 0.09 mm after 60 s). This
  is the regression test for model mismatches between the simulator and the estimator. During
  development it exposed a real one: `propagate()` took its first IMU sample by const
  reference to a member it then overwrote, so every step integrated the *end* sample twice.
  That is a hidden 2.5 ms IMU-camera time shift, worth ~2 mrad of attitude error at
  1 rad/s, and it showed up as a systematic, seed-independent 0.3° extrinsic error during
  convergence. `tests/test_calibrator.cpp` now pins propagation accuracy and its O(dt²)
  convergence order.

## 9. Self-assessment

`ExtrinsicCalibrator::status()` turns the covariance into something an operator, or a
supervising process, can act on:

* **Bounds along the least-certain directions.** The rotation and lever-arm blocks of `P` are
  eigen-decomposed, and the bound is `k σ_max` along the largest eigenvector (`k = 2` by
  default). A single badly observed direction then cannot hide behind two good ones, as it
  would in a per-axis or trace-based summary. The directions are reported as unit vectors in
  the IMU frame. The rotation error is a camera-frame rotation vector, so its axis is rotated
  by `R_IC`.
* **State.** *Converged* when the rotation and lever-arm bounds are inside 0.5° / 1 cm (the
  acceptance criteria) and the time-offset bound is inside 1 ms.
* **Why 2σ, and what it does not promise.** The bounds are `2 σ_max` of the filter, not
  guarantees. Even a consistent filter would see the error norm beyond `2 σ_max` now and then:
  for an isotropic 3-D Gaussian, `P(‖e‖ > 2σ) = P(χ²₃ > 4) ≈ 0.26`. The *converged* verdict
  only asks for the bounds to be inside the 0.5° / 1 cm tolerance, and 2σ is enough for that
  because the covariance is conservative (NEES ≈ 2 against 6, §8): a filter σ is about 1.8
  true σ. In the evaluation (README), it never said *converged* while the true error was
  outside tolerance in the 200 independent runs where the model holds and the mount is still
  (`convergence`, `time_offset`, `outliers`, `noise_misspecified`), nor in the 10 min of drift
  that the `drift` runs add after the same first minute. Against its own bounds the
  record is weaker, as it should be for 2σ. The true error was outside them on 0.04 % of the
  converged frames on a still mount, and on 1.3 % during the drift (the random walk lags it
  slightly).
* **How cautious.** In the nominal scenarios it certifies about 10 s after the true error gets
  inside (16.3 s against 6.0 s, p50). With 2× pixel noise it certifies about 37 s after
  (45.6 s against 8.1 s), because the larger noise keeps the covariance wide. A 3σ bound
  certified at 34 s and, with 2× pixel noise, not at all within 60 s, for no gain in safety.
* **The 2σ choice leans on the conservative covariance.** With disjoint frame pairs (NEES 5.3,
  close to consistent), the same 2σ status made 240 false claims in 50 runs, and the true error
  was outside its bounds on 12 % of the converged frames. A consistent covariance (a sliding
  window, so each observation is used once) would need a larger multiplier.
* **Recalibrating.** A detected knock sets the state to *recalibrating* until all three bounds
  are inside tolerance again. In the evaluation, the frames between a knock and its detection
  were the only false claims with the model otherwise right.
* **Advice.** A direction `a` of the lever arm, or an axis `a` of the rotation, is invisible to
  rotation about `a` itself and observed only by rotation about the perpendicular axes. The
  filter keeps a running second moment `M` of the body rate (5 s time constant), so the
  rate available to observe `a` is `√(tr M − aᵀMa)`. Below 0.1 rad/s the status says the
  direction *needs motion* and names the two rotations that would observe it; below 0.3 rad/s
  it says convergence is *slow* and which rotations would speed it up; otherwise it reports
  the remaining bound.

What it cannot see: anything the model does not contain. With the timing model wrong
(`time_offset_ignored`), with the noise model wrong and not estimated
(`noise_misspecified_fixed`), or with the mount moving faster than the random walk allows and
no detector (`drift_frozen`, `knock_no_detector`), the covariance is wrong and so is the
verdict. The evaluation shows these false claims explicitly, next to the zero for the
scenarios where the model holds.
