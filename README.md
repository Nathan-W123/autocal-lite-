# AutoCal-lite

**Targetless, online stereo/IMU extrinsic calibration that recovers a rig from a 5° / 5 cm
miscalibration in seconds and keeps tracking the mount as it drifts.** It is an error-state EKF
in C++17, built on the estimation, sensor-model and Monte-Carlo core of
[Aether6](https://github.com/Nathan-W123/Aether6). It uses no calibration target, no operator
and no ground truth, only the sensor data.

![Convergence over 50 runs](results/figures/convergence.png)

## Results

50 Monte-Carlo runs per scenario, with deterministic seeds: `make eval` reproduces every
number below with the same standard library, here libstdc++ (GCC 13 and Clang 18 give identical
runs). `std::normal_distribution` is implementation-defined, so another standard library gives
statistically equivalent but not identical numbers. Each run draws its own landmark field,
truth mount (nominal ± 2° / ± 2 cm), IMU biases and noise. It also draws the *direction* of the 5° / 5 cm error in the starting
estimate: the starting extrinsic is the hidden truth perturbed by exactly 5° and 5 cm, and it
is the only thing the estimator receives.

| criterion | target | result |
|---|---|---|
| **1. Converge** from 5° rotation / 5 cm translation error | < 0.5° and < 1 cm within 60 s in ≥ 95 % of 50 runs | **50 / 50 runs (100 %)**. Error at 60 s, p95: **0.025° / 0.34 cm** (worst run 0.029° / 0.40 cm). Median time to converge 6.0 s (p95 9.3 s) |
| **2. Track drift**: after convergence the true mount rotates at 0.1°/min for 10 min (1° total) | tracking error < 0.3° | **50 / 50 runs**. Worst tracking error over the 10 min: **0.061°** (p95 0.056°, RMS 0.024°). The same filter with the random walk and the change detector switched off falls behind: 0.55° after 10 min and still diverging (0 / 50 pass) |

![Drift tracking](results/figures/drift_tracking.png)

Supplementary scenarios (same 50 seeds):

| scenario | what changes | result |
|---|---|---|
| `time_offset` | camera clock 8 ms off the IMU clock (t_d is estimated in every scenario) | 50 / 50 converge; t_d recovered to 0.03 ms (p50), 0.07 ms (p95) |
| `time_offset_ignored` | camera lags 2.5 ms but the filter assumes perfect sync | **21 / 50 converge**: the lever arm absorbs the lag (1.1 cm p50, 3.5 cm p95 error). The change detector flags the misfit, firing 3× per run, and the status certifies none of the results at 60 s. This is why t_d is always estimated |
| `noise_misspecified` | pixels are twice as noisy as configured (1.0 px vs 0.5 px) | 50 / 50 converge (p95 0.027° / 0.52 cm): the filter measures the real noise online (1.005 px, p50) and its covariance stays conservative (NEES 1.4) |
| `noise_misspecified_fixed` | same, noise estimation switched off | **33 / 50 converge** (p95 0.075° / 1.39 cm), NEES 23: confidently wrong |
| `outliers` | 5 % of the correspondences are wrong | 50 / 50 converge (p95 0.027° / 0.29 cm); the per-landmark χ² gate removes them |
| `knock` | the mount is knocked by 1.5° / 1.95 cm at t = 90 s | the change detector fires exactly once in every run, 0.25 s after the knock (p50; 0.40 s worst). Back under 0.5° and 1 cm after 2.6 s (p50) / 5.1 s (p95), 0.012° / 0.12 cm at 180 s. While it re-converges the lever arm overshoots to 2.9 cm (p50), and in 13 runs the rotation briefly exceeds the knock (2.0° worst) |
| `knock_no_detector` | same knock, detector off | the random walk recovers the rotation but not the lever arm: still 1.05 cm off at 180 s (11 / 50 back under 1 cm) |
| `yaw_only` | roll and pitch excitation removed | 5 / 50 converge: the lever arm along the yaw axis is unobservable (3.1 cm p50 error), exactly as the theory predicts |
| false alarms | change detector in every run where it is armed and the model is right (`drift`, `time_offset`, `outliers`, `noise_misspecified`) | **0 in 200 runs** (≈ 11.7 h of simulated data), and none before the knock in the 50 `knock` runs. In `yaw_only` it never arms, because the filter knows its lever arm has not converged |

Every frame is delivered to the estimator 30 ms after its exposure; no frame was dropped in
any run. The extrinsic covariance is reported alongside every estimate. The pixel noise is
estimated online (0.503 px p50 for 0.5 px), so the noise model keeps matching the data when the
configured value is wrong, and the state-dependent vision innovations have NIS 0.99 per degree
of freedom. The extrinsic NEES is 1.9 (6 = consistent): the filter is **conservative** by about
√(6/1.9) ≈ 1.8× in σ, for a known reason ([docs/method.md §8](docs/method.md#8-consistency-honestly)).
A 60 s run takes about 1.4 s of wall-clock time on one core (about 0.2 s of it is the noise
estimate).

### Does it know when it is done?

`ExtrinsicCalibrator::status()` turns the covariance into a verdict an operator or a
supervising process can act on. It reports a state (initialising / calibrating / converged /
recalibrating), 2σ bounds along the *least-certain* rotation and lever-arm directions, which
directions those are, the recent rotation excitation about each IMU axis, and one line of
advice. *Converged* means both bounds are inside the 0.5° / 1 cm acceptance criteria. A *false
claim* below is a frame on which it says *converged* while the true error is outside that
tolerance.

| scenario | says converged at 60 s | certified at (p50) | truth inside at (p50) | false claims (frames) | true error beyond its own 2σ bounds |
|---|---|---|---|---|---|
| `convergence`, `time_offset`, `outliers` | 150 / 150 | 16.3–17.0 s | 6.0–6.4 s | **0** | 0–0.04 % of converged frames |
| `noise_misspecified` (2× pixel noise, estimated) | 50 / 50 | 45.6 s | 8.1 s | **0** | 0 % |
| `drift` (10 min at 0.1°/min) | 50 / 50 | 16.3 s | 6.0 s | **0** | 1.3 % (the random walk lags the drift slightly) |
| `yaw_only` | 0 / 50 (correct: the lever arm is not observable) | – | – | **0** | – |
| `knock` (1.5° / 1.95 cm at 90 s) | 50 / 50 | 16.3 s | 6.0 s | 253, all between a knock and its detection (0.25 s p50); none after | 0.18 % |
| `noise_misspecified_fixed` (noise estimation off) | 49 / 50 | 22.8 s | 29.3 s | 15 166: why the noise must be estimated | 82 % |
| `time_offset_ignored` (timing model wrong) | 0 / 50 | – | – | 455 | 50 % |

Wherever the model holds, *converged* has meant inside tolerance on every frame: 0 false claims
in 200 independent runs, and none in the 10 min of drift that the `drift` runs add. It is
cautious: in the nominal scenarios it certifies about 10 s after the truth gets inside, and with 2× pixel noise about
37 s after. The bounds themselves are 2σ of the filter, not guarantees. 2σ is enough for the
verdict only because the covariance is conservative: with a near-consistent covariance
(disjoint frame pairs, below) the same status made 240 false claims
([docs/method.md §9](docs/method.md#9-self-assessment)). Where the model is wrong (noise not
estimated, timing ignored, or, in the evaluation, a drifting mount without the random walk or a
knock without the detector), the covariance is wrong and so is the verdict, and the evaluation
shows it. The advice tracks what the motion can and cannot observe:

```
converged:   calibrated: rotation +-0.042 deg, lever arm +-4.9 mm (2-sigma)
gentle:      converging slowly: lever arm along IMU y (right) +-23.8 mm; more yaw (about z) or roll (about x) would speed it up
yaw only:    needs motion: lever arm along IMU z (down) is not observable from the current motion; roll (about x) or pitch (about y) the rig
```

(`autocal_run --scenario convergence`, the same with `--set roll_pitch_scale=0.2 --set
yaw_scale=0.5`, and `--scenario yaw_only`; trial 0, status at 60 s.)

### Sensitivity checks

Same 50 seeds, one setting changed; `make eval` runs these too (`results/mc/sensitivity/`).

| check | result |
|---|---|
| Pixel noise configured 10× too low or 10× too high (0.05 or 5 px for 0.5 px), with or without 5 % wrong correspondences | 50 / 50 converge in all four cases, with unchanged accuracy (p95 ≤ 0.027° / 0.35 cm). 0 false claims. The noise is estimated as 0.491–0.513 px |
| Status at 3σ instead of 2σ | Certifies at 34.1 s instead of 16.3 s (p50). With 2× pixel noise, it never certifies within 60 s (0 / 50). No safety gained: 2σ already made no false claims there |
| Disjoint frame pairs (each observation used once) | NEES 5.3, close to consistent, but twice the error (p95 0.045° / 0.73 cm). The 2σ status then makes 240 false claims |
| 5° / 5 cm knock | Detected once per run. Back in tolerance after 3.6 s (p50) / 7.7 s (p95). 75 false claims, all before detection. With the earlier 1° / 1 cm re-open: 8.2 s, and 283 false claims after detection |
| Knock re-open without loosening the navigation states | 135 false claims after detection for the 1.5° knock, and 320 for the 5° knock (none with the loosening) |

## The problem

A stereo camera and an IMU are bolted to the same rig. Fusing them needs the rigid transform
between them: the extrinsic, 3 rotation + 3 translation parameters. Today that transform is
measured once with a checkerboard and an expert, and measured again whenever the hardware
shifts: a knock, thermal cycling, a re-mount. AutoCal-lite estimates it **continuously from
normal operation**:

* **Input.** 200 Hz IMU samples and 20 Hz stereo frames with feature correspondences (the
  front end is out of scope; correspondences are provided, 0.5 px noise). Known intrinsics and
  a 12 cm baseline. A nominal ("CAD") extrinsic that is wrong by 5° and 5 cm (generated per
  run as described above).
* **Not input.** Landmark positions (there is no target), the trajectory, the true extrinsic,
  or any other ground truth. The estimator initialises itself from a 1.5 s stationary IMU
  window.
* **Output.** The extrinsic with a covariance, the IMU biases and the camera-IMU time offset,
  updated at every frame. The time offset is estimated in every scenario, even when its true
  value is 0, because the lever arm is very sensitive to timing: an unmodelled 2.5 ms camera
  lag alone biases it by about 1 cm (`time_offset_ignored` below).
* **Delivery.** Each stereo frame reaches the estimator 30 ms after its exposure, as from a
  camera driver. The filter runs 80 ms behind the newest IMU sample, so every frame is still
  fused at its exact exposure time.

## How it works

```
 IMU 200 Hz ──► trapezoidal propagation of pose/velocity/biases (Aether6 error dynamics)
                          │
 stereo 20 Hz ─► landmarks seen in the previous AND current frame
                 ├─ triangulate (stereo) + 3 Gauss-Newton steps over the 4 views
                 ├─ 8 reprojection residuals per landmark, predicted with the CURRENT
                 │  extrinsic estimate; analytic Jacobians
                 ├─ landmark marginalised exactly (left null space, 5 residuals remain)
                 ├─ chi-square gate per landmark (wrong correspondences)
                 └─ QR-compress ~500 rows to 28 ─► Aether6 Joseph-form update
                          │
            stochastic clone moves to the current pose; change detector watches the
            extrinsic corrections and re-opens its covariance after a knock
```

* **Small state, no landmarks.** 28 error states: IMU pose/velocity (9), biases (6),
  extrinsic (6), time offset (1), and one stochastic clone of the previous frame's pose (6).
  The landmarks are marginalised per update, so there is no map and no sliding window.
* **Never recalibrate.** The extrinsic is a random walk. Its intensity was chosen by a sweep
  so that a drifting mount is followed with a lag far inside the requirement, while the
  covariance stays consistent with the drift. A change detector catches knocks.
* **Observability.** The figure-eight trajectory carries roll, pitch *and* yaw excitation.
  With rotation about a single axis, the lever-arm component along that axis is
  unobservable (the `yaw_only` row above shows this).
* **Self-tuning noise.** Projected off every state direction, the stacked residual is pure pixel
  noise with a known number of degrees of freedom. A robust (median) estimate of its level,
  taken before the gate and without using the filter's covariance, rescales the pixel noise
  model at every frame.
* **Self-assessment.** `status()` reads the covariance: 2σ bounds and weakest directions,
  excitation per axis, and advice on which rotation would make the weakest direction
  observable.

The full derivation (Jacobians, null-space marginalisation, observability, the drift/knock
logic, and a candid consistency analysis) is in [docs/method.md](docs/method.md).

## Built on Aether6

The tree mirrors [Aether6](https://github.com/Nathan-W123/Aether6)'s layout (`include/aether/…`,
`src/…`, `apps/`, `tests/`, same CMake/Makefile/CI and namespaces), so the module is laid out
to be merged back. It is not a copy-in drop, though. Merging it into the Aether6 tree means:

* adding `calib/` and `estimation/EskfCore`;
* making `sensors/Imu.hpp` include `ImuConfig`/`ImuSample` from Aether6's
  `sensors/Sensors.hpp` instead of redefining them (Aether6's `SensorSuite` could then delegate
  its IMU branch to `ImuModel`);
* adding `mc::runCampaign`/`trialSeed` to Aether6's existing `mc/MonteCarlo.hpp`, whose
  aircraft-specific runner keeps the same file;
* adding the `Vec2` alias to `core/Types.hpp`.

Files taken from Aether6 (MIT, commit `6b8dc56`) say so in their header.

| | what | where |
|---|---|---|
| **Reused** | Quaternion/rotation algebra, Van Loan discretisation, matrix exponential | `math/Rotation.hpp`, `math/LinearAlgebra.*` (verbatim; LQR/Riccati dropped) |
| **Reused** | Central-difference Jacobians (used by the Jacobian tests) | `math/NumericalDiff.hpp` (verbatim) |
| **Reused** | SplitMix64 seed derivation, deterministic RNG, CSV writer | `util/Random.hpp`, `util/Csv.*` (verbatim, except that `gaussian3()` now sequences its three draws: the argument evaluation order it relied on differs between GCC and Clang) |
| **Reused** | IMU white noise + initial bias + bias random walk, same RNG stream | `sensors/Imu.*` (Aether6 `SensorSuite` IMU branch, lifted out) |
| **Reused** | ESKF numerics: series/Van Loan discretisation, gated Joseph-form update, multiplicative attitude reset | `estimation/EskfCore.*` (extracted from Aether6 `ErrorStateEkf`) |
| **Reused** | Deterministic multi-threaded Monte Carlo and its statistics | `mc/MonteCarlo.*` (thread pool + seed scheme generalised to a template; `computeStatistics` verbatim) |
| **Reused** | Build system and CI (system-or-fetched Eigen/Catch2, strict warnings, `-Werror` CI on GCC/Clang/Debug) | `CMakeLists.txt`, `Makefile`, `.github/workflows/ci.yml` |
| **Adapted** | Error-state EKF: Aether6's 18-state nav filter becomes a 28-state calibrator (extrinsic + time offset + clone; wind/baro states removed) | `calib/ExtrinsicCalibrator.*` |
| **Replaced** | Flight dynamics replaced by a scripted figure-eight with roll/pitch/yaw excitation and analytic derivatives | `calib/Trajectory.*` |
| **New** | Stereo camera + landmark scene + truth mount with drift/knock; reprojection model; trial scoring and scenarios | `calib/StereoCamera.*`, `calib/CalibSimulator.*`, `calib/MeasurementModel.*`, `calib/Campaign.*` |

## Reproduce

Prerequisites: a C++17 compiler (tested with GCC 13), CMake ≥ 3.16, and Python 3.9+
(standard library only; Matplotlib optional, for the figures). Eigen 3 and Catch2 v3 are
taken from the system when present and **fetched automatically otherwise**. There is no ROS
dependency.

```bash
make build      # configure + compile (Release)
make test       # Catch2 suite
make run        # one 60 s convergence run: prints its scores, writes results/runs/
make eval       # 50-run campaigns for every scenario and sensitivity check (~15 min on 4
                # cores), then prints every table above and the drift-tracking plot data,
                # and renders results/figures/
```

`python3 python/scripts/eval.py --skip-run` re-prints the tables from the committed
`results/mc/` (including the sensitivity campaigns in `results/mc/sensitivity/`) without
re-running anything. The apps take `--help`. The parameters listed in
`calib::applyOverride` (`include/aether/calib/Campaign.hpp`) can be overridden from the
command line, for example:

```bash
./build/bin/autocal_run --scenario drift --set rot_walk=5e-5
./build/bin/autocal_mc  --scenario convergence --trials 50 --set pixel_noise=1.0 --set pixel_sigma=1.0
```

Scenarios: `convergence`, `drift`, `drift_frozen`, `knock`, `knock_no_detector`,
`time_offset`, `time_offset_ignored`, `noise_misspecified`, `noise_misspecified_fixed`,
`yaw_only`, `outliers`. `autocal_run --trial I`
reproduces trial `I` of a campaign exactly.

## Layout

```
include/aether/calib/   the calibration module (new)       src/calib/      its sources
include/aether/{core,math,util,sensors,estimation,mc}/      reused Aether6 core
apps/autocal_run.cpp    one trial -> scores + time series
apps/autocal_mc.cpp     deterministic Monte-Carlo campaigns -> trials.csv, envelope.csv, summary.json
python/scripts/eval.py  criteria tables, drift-tracking plot data, figures
tests/                  Catch2 suite (Jacobians vs. numerical, simulator truth, filter regressions)
docs/method.md          derivation and design notes
results/                committed campaign outputs and figures
```

## What this demonstrates for AutoCal

AutoCal's hard problem is an extrinsic calibration that stays right without targets or
experts. This prototype runs that whole loop end to end and measures it. The scene itself is
the target: landmarks are triangulated and marginalised on the fly and never surveyed. The
rig initialises itself from 1.5 s of stillness and pulls a CAD-level 5° / 5 cm error below
0.03° / 5 mm in every one of 50 randomised runs (median 0.012° / 1.5 mm), typically
converging within 6 s. Because the extrinsic is modelled as a slowly varying state with a
covariance, not a constant measured once, the same filter holds a mount that drifts 1° in
10 minutes to within 0.06°, where a filter that treats the mount as fixed ends 0.55° off and
keeps diverging. When the hardware is knocked, a χ² test on the filter's own corrections
notices within 0.4 s, and the calibration re-converges in seconds with no operator. That is
the "never recalibrate again" behaviour, in miniature.

It also knows when it is done. It measures its own pixel noise, even when the configured value
is 10× off. It certifies convergence only when its bounds are inside tolerance: not once
wrongly in 200 independent runs where the model holds, including 50 with misconfigured noise
that it corrected itself, and not while the mount drifted for 10 min. When the motion cannot observe part
of the extrinsic, it says which part and which rotation would fix it.

What makes it trustworthy rather than a demo is what AutoCal needs at scale: analytic
Jacobians checked against numerical ones, a noise-free regression test that catches
simulator/estimator timing mismatches, consistency statistics reported next to accuracy, a
self-assessment scored against the hidden truth, zero false change alarms across 200 armed
runs, and an observability ablation that shows which motions the calibration needs. Growing
it toward AutoCal means adding states and sensors (intrinsics, more cameras, LiDAR), not
changing the approach.

## Scope and limitations

This is a prototype on simulated data, not a product. The camera is an ideal rectified
pinhole pair: no lens distortion, no rolling shutter, known intrinsics. The pixel noise is
estimated online, but the IMU noise parameters (white-noise densities and bias random walks)
are the simulator's true values; with real sensors they would come from an Allan-variance
characterisation and would be inflated somewhat. Correspondences are given rather than
detected. Only stereo + IMU is calibrated. The motion is a deliberately exciting figure-eight. The filter's covariance is
conservative by roughly 1.8× in σ ([docs/method.md §8](docs/method.md#8-consistency-honestly)
explains why and what it would cost to remove), and the status's 2σ threshold relies on that
margin. The natural next steps toward AutoCal are longer feature tracks (a short sliding
window, for more information per update under gentle motion and a consistent covariance; the
status multiplier would then need re-tuning), camera intrinsics in the state, real data with
a feature front end, and more sensor pairs (LiDAR-camera, multi-camera).
