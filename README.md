# Differential Drive Slam

EKF-SLAM with landmark association and occupancy grid mapping for a simulated differential drive robot.

[![CI](https://github.com/Eelis03/differential-drive-slam/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/differential-drive-slam/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This library estimates the pose of a differential drive robot and the positions of
the landmarks around it from noisy wheel commands and noisy range and bearing
detections, using an extended Kalman filter over the joint robot and map state. It
also builds a log-odds occupancy grid from simulated laser scans taken at the
filtered pose. A simulator with full ground truth, a metrics module covering
absolute trajectory error, landmark RMSE, and normalised estimation error squared,
and four runnable example scripts are included, so every number in the Results
section can be reproduced from a single command.

It is aimed at anyone who needs a small, readable, dependency-light reference
implementation of classical feature-based SLAM: for teaching, for checking a larger
system against a known-good baseline, or as a starting point for experiments with
data association and filter consistency.

## Problem

A wheeled robot moving through an unknown environment accumulates unbounded error if
it integrates its wheel commands alone. Over the 64 second run used throughout this
repository, dead reckoning drifts by 0.51 m RMSE and 1.00 m at worst, and the drift
grows without limit as the run continues.

Correcting that drift requires a map, but building a map requires knowing where the
robot is. The two problems have to be solved together. The estimator must therefore:

1. Propagate a joint belief over the robot pose and every landmark position through a
   nonlinear differential drive motion model, keeping the correlations between the
   robot and the map, since those correlations are what allow a revisited landmark to
   correct the pose.
2. Decide, for every incoming range and bearing detection, which mapped landmark it
   came from, or whether it came from a landmark not yet in the map. Detections carry
   no identity, so this is a decision under uncertainty, and a single wrong assignment
   can corrupt the whole map.
3. Report an uncertainty that is neither too large to be useful nor too small to be
   trusted. A filter whose reported covariance is smaller than its actual error is
   dangerous downstream, so consistency must be measured rather than assumed.
4. Turn the range measurements into a dense representation of free and occupied space
   that a planner can use, given that the poses those measurements were taken from are
   themselves estimates.

The simulated setting fixes the world frame and provides ground truth, which makes
these requirements measurable: trajectory error needs no alignment step, and landmark
error can be scored against the landmark that actually generated each detection.

## Approach

The estimator is EKF-SLAM in the stochastic map formulation of Smith, Self, and
Cheeseman, following the presentation in chapter 10 of Probabilistic Robotics. A
single Gaussian is maintained over the state `[x, y, theta, m1x, m1y, ...]` with a
dense covariance. Prediction propagates the robot block through the exact arc solution
of the differential drive kinematics and mixes the added process noise into the robot
to map cross covariances, with the control covariance scaled by the square of the
commanded velocities as in the velocity motion model. Correction applies one range and
bearing measurement at a time in Joseph form, which keeps the covariance symmetric and
positive semidefinite. A landmark is appended to the state on its first observation,
with a covariance obtained by pushing the pose uncertainty and the measurement
uncertainty through the inverse sensor model Jacobians, so the new landmark starts out
correlated with the robot and with the rest of the map.

Data association is maximum likelihood under a chi-square gate. The squared
Mahalanobis distance of each candidate innovation is chi-square distributed with two
degrees of freedom when the association is correct, which sets a principled acceptance
threshold; among the candidates that pass the gate, the one with the highest Gaussian
likelihood is chosen, so the innovation covariance normaliser is retained rather than
dropped. A second, looser threshold decides when a detection is far enough from
everything in the map to initialise a new landmark, and detections that fall between
the two are discarded as ambiguous rather than being forced into one of the two
decisions.

Mapping uses the recursive log-odds occupancy grid of Moravec and Elfes. Each beam
lowers the log odds of the cells it passes through and raises the log odds of the cell
it terminates in, with the accumulated value clamped so that no cell becomes so certain
that later evidence cannot revise it.

EKF-SLAM was chosen because it is the formulation in which the correlation structure of
the problem, the cost of maintaining it, and the consistency question are all visible in
the code rather than hidden behind a solver. The alternatives that were considered and
rejected, including FastSLAM, graph-based smoothing, and the unscented filter, are
recorded in [docs/design-notes.md](docs/design-notes.md) together with the known
limitations, of which the quadratic growth of the covariance in the landmark count is
the most important.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/differential-drive-slam.git
cd differential-drive-slam
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```python
from diffdrive_slam import SimulationConfig, evaluate, run_simulation

trace = run_simulation(SimulationConfig(steps=640, seed=20260731, build_grid=False))
result = evaluate(trace)

print(f"SLAM  ATE RMSE     {result.trajectory.position_rmse:.4f} m")
print(f"odometry ATE RMSE  {result.dead_reckoning.position_rmse:.4f} m")
print(f"landmark RMSE      {result.landmarks.rmse:.4f} m")
print(f"landmarks          {result.landmarks.estimated} of {trace.true_landmarks.shape[0]}")
```

Output:

```text
SLAM  ATE RMSE     0.0541 m
odometry ATE RMSE  0.5146 m
landmark RMSE      0.0275 m
landmarks          20 of 20
```

Runnable examples live in `examples/`:

```bash
uv run python examples/run_ekf_slam.py
uv run python examples/run_occupancy_grid.py
uv run python examples/run_consistency_study.py
uv run python examples/run_data_association.py
```

Every script accepts `--steps` and `--seed`, and the three that draw figures accept
`--output` and `--no-figures`.

## Results

All numbers below are the output of the commands shown, produced on Python 3.12 with
numpy 2.5.1, scipy 1.18.0, and matplotlib 3.11.1.

### Configuration

The default scenario is a square room 11 m on a side with a 2.4 m block obstacle at its
centre and 20 point landmarks placed by rejection sampling with a minimum separation of
1.0 m. The robot drives a closed loop of four 5.0 m straight legs joined by quarter
arcs, at 1.0 m/s with 0.785 rad/s turns, giving a path 7.5 m across that is centred in
the room. The run is 640 steps of 0.1 s, which is 2.3 laps. Control noise uses the
coefficients `(0.010, 0.002, 0.002, 0.010)`, giving a velocity standard deviation near
0.10 m/s at 1.0 m/s. The landmark detector has a range standard deviation of 0.15 m, a
bearing standard deviation of 0.026 rad, a maximum range of 4.0 m, and a full 360 degree
field of view. The laser has 45 beams, a maximum range of 6.0 m, and a range standard
deviation of 0.05 m, and is integrated every second step. Gates are set at the 0.99
chi-square quantile for acceptance and the 0.9999 quantile for initialising a landmark.

### Localisation and mapping accuracy

`uv run python examples/run_ekf_slam.py`

```text
steps                        640
true landmarks               20
estimated landmarks          20
ATE position RMSE [m]        0.0541
ATE position max [m]         0.1598
ATE heading RMSE [rad]       0.0101
dead reckoning RMSE [m]      0.5146
dead reckoning max [m]       1.0021
landmark position RMSE [m]   0.0275
landmark position max [m]    0.0498
measurements                 4040
matched                      3976
initialised                  20
rejected as ambiguous        44
incorrect matches            0
association accuracy         1.0000
time averaged NEES           1.0708
expected value               3
per step bounds (95 percent) [0.2158, 9.3484]
steps inside per step bounds 0.7941
nominal inside fraction      0.9500
pooled bounds (95 percent)   [2.8133, 3.1926]
verdict on pooled average    conservative
```

On this seed the filter reduces the trajectory error by a factor of 9.5 relative to dead
reckoning, recovers all 20 landmarks with no duplicates, and makes no incorrect
association across 4040 detections. The 44 detections rejected as ambiguous are 1.1
percent of the total. This seed is a favourable draw: its trajectory error of 0.054 m is
about half the 0.114 m mean reported by the Monte Carlo study below, so the single run
figures should be read alongside the ensemble ones and not in place of them.

### Filter consistency

`uv run python examples/run_consistency_study.py`

Twenty independent noise realisations over the same map, with the NEES averaged across
runs at each time step:

```text
runs                         20
steps per run                640
ATE position RMSE mean [m]   0.1141
ATE position RMSE std [m]    0.0705
landmark RMSE mean [m]       0.1250
landmark RMSE std [m]        0.0679
ensemble average NEES        2.9066
expected value               3
per step bounds (95 percent) [2.0241, 4.1649]
steps inside per step bounds 0.9750
nominal inside fraction      0.9500
pooled bounds (95 percent)   [2.9577, 3.0425]
verdict on pooled average    conservative
```

The verdict is that the filter is consistent, marginally on the conservative side.

The per-step test is the one that carries the evidence. At each time step the average of
20 independent NEES values should lie inside [2.0241, 4.1649] with probability 0.95 if
the filter is consistent. 97.5 percent of the 641 time steps do, against a nominal 95
percent, so the reported covariance is not too small anywhere along the trajectory. The
ensemble average NEES over the whole run is 2.9066 against an expected value of 3, which
is 3.1 percent low. The pooled interval [2.9577, 3.0425] places that just outside, hence
the printed verdict of conservative, but the pooled interval treats the 640 time samples
within a run as independent when they are strongly correlated, so it is far tighter than
the evidence supports and is printed for completeness rather than used as the test.

Reading both together: the filter reports very slightly more uncertainty than it has,
and nowhere reports less. That is the safe direction of error.

This result should not be read as a general statement about EKF-SLAM, which is known to
become optimistic when the heading uncertainty grows large between loop closures. The
scenario here keeps the heading uncertainty small, with a heading RMSE of 0.010 rad, so
it does not exercise that regime. See [docs/design-notes.md](docs/design-notes.md) for
the conditions under which the verdict is expected to change.

The single run figures reported by `run_ekf_slam.py` show a time averaged NEES of 1.07,
well below the ensemble figure. A single 640-step run is not 640 independent samples, so
that number carries much less information than the ensemble average and is reported only
alongside the caveat.

### Data association

`uv run python examples/run_data_association.py`

Five noise realisations, each run twice, once with the true correspondences supplied to
the filter and once with the correspondences recovered by the maximum likelihood policy:

```text
steps per run                640
seeds                        5
association           ATE RMSE [m]   landmark RMSE [m]   landmarks  incorrect
known                       0.1013              0.0937        20.0          0
maximum likelihood          0.1007              0.1052        20.4          0
measurements (ML runs)       20407
rejected as ambiguous        190
rejection rate               0.0093
```

Across 20407 detections the policy makes no incorrect assignment. The difference in
trajectory error between the two modes, 0.1007 m against 0.1013 m, is far smaller than
the run to run standard deviation of 0.0705 m reported by the consistency study, so on
this scenario solving the association problem from the measurements alone costs nothing
measurable in localisation accuracy. It does cost in map quality: landmark RMSE rises
from 0.0937 m to 0.1052 m, and the maximum likelihood runs end with 20.4 landmarks on
average against a true count of 20, so roughly one spurious landmark is created every
two or three runs and never merged. It also discards 0.93 percent of detections as
ambiguous.

### Occupancy grid

`uv run python examples/run_occupancy_grid.py`

```text
steps                        640
grid                         120 x 120 cells at 0.10 m
beams per scan               45
scan interval [steps]        2
log odds bounds              [-4.0, 4.0]
ATE position RMSE [m]        0.0541
cells                        14400
classified occupied          1082
classified free              11105
unknown (prior retained)     2213
decided fraction             0.8463
free agreement               0.9902
occupied agreement against the wall tolerance:
  tolerance 0 cells (0.00 m)  0.3530
  tolerance 1 cells (0.10 m)  0.8586
  tolerance 2 cells (0.20 m)  0.9972
  tolerance 3 cells (0.30 m)  1.0000
```

84.6 percent of cells are decided; the remaining 2213 keep the prior, and are the cells
behind the walls, inside the obstacle, and in the 0.5 m margin between the wall and the
grid boundary, none of which any beam reaches. Of the cells the grid calls free, 99.0
percent are genuinely free. Of the cells it calls occupied, 35.3 percent land exactly on
a wall cell, 85.9 percent land within one cell of a wall, and 99.7 percent within two
cells. The tolerance sweep is reported rather than a single number because the two error
sources have different scales: the walls are infinitely thin in the simulated world while
the grid quantises them into 0.10 m cells, and the map is built at the filtered pose,
which carries its own error. A single occupancy figure at a fixed tolerance would hide
which of those dominates.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/diffdrive_slam/model/arrays.py` | Array aliases, angle wrapping, covariance symmetrisation, positive semidefiniteness checks |
| `src/diffdrive_slam/model/motion.py` | Differential drive kinematics, wheel rate conversion, velocity noise model, Jacobians with respect to state and control |
| `src/diffdrive_slam/model/sensor.py` | Range and bearing measurement model, visibility test, inverse model, both sets of Jacobians |
| `src/diffdrive_slam/model/state.py` | Joint robot and map Gaussian, landmark indexing, marginal extraction |
| `src/diffdrive_slam/model/grid.py` | Grid geometry, world to cell mapping, log-odds conversions, line rasterisation |
| `src/diffdrive_slam/algorithm/ekf_slam.py` | Prediction, Joseph form correction, state augmentation, batch measurement integration |
| `src/diffdrive_slam/algorithm/association.py` | Mahalanobis distance, chi-square gating, maximum likelihood selection, association outcomes |
| `src/diffdrive_slam/algorithm/occupancy.py` | Log-odds inverse sensor model, beam and scan integration, clamping |
| `src/diffdrive_slam/pipeline/environment.py` | Landmarks and walls, visibility queries, ray casting, ground truth grid rasterisation |
| `src/diffdrive_slam/pipeline/trajectory.py` | Open-loop control sequences: closed square loop and figure eight |
| `src/diffdrive_slam/pipeline/simulate.py` | The run loop: noisy control and measurement generation, filter driving, grid mapping |
| `src/diffdrive_slam/pipeline/trace.py` | The structured record of a run, with ground truth and per-step association detail |
| `src/diffdrive_slam/analysis/metrics.py` | Trajectory error, landmark RMSE, NEES and chi-square bounds, association and grid scoring |
| `src/diffdrive_slam/analysis/figures.py` | Trajectory, error history, NEES, and occupancy grid figures |
| `examples/run_ekf_slam.py` | Wiring: one run, accuracy and consistency report, three figures |
| `examples/run_occupancy_grid.py` | Wiring: one run with mapping, grid scoring against ground truth, one figure |
| `examples/run_consistency_study.py` | Wiring: Monte Carlo NEES study over independent noise seeds |
| `examples/run_data_association.py` | Wiring: known correspondences against maximum likelihood association |

The dependency direction is strictly `model` to `algorithm` to `pipeline` to `analysis`
to `examples`. The model layer holds pure functions and dataclasses with no I/O and no
state. The algorithm layer contains estimation only: it draws no random numbers and
produces no plots. The pipeline layer is the only place random numbers are drawn. The
analysis layer reads traces and produces numbers and figures. The example scripts
contain wiring and printing, and no logic that is not tested elsewhere.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each example
script under a reduced iteration count.

The first tier checks that the motion model integrates a straight line and a pure
rotation exactly, that a full circle returns to its starting point, that every analytic
Jacobian matches a central finite difference to 1e-6, that the covariance stays symmetric
and positive semidefinite over a full run, that observing a landmark reduces its marginal
uncertainty monotonically, that the filter converges to ground truth within 0.05 m under
known correspondences, that log-odds updates saturate at the configured bounds, and that
an unobserved cell keeps its prior. The second tier replays a recorded 150-step run and
compares poses, covariances, landmark positions, metrics, association counts, and the
grid against `tests/data/reference_run.json` with a tolerance of 1e-6; regenerate it with
`uv run python tests/generate_reference.py` when a change to the algorithm is intended.
The third tier runs all four example scripts as subprocesses under reduced step counts
and checks that the figure writing paths also work.

182 tests run in about 11 seconds.

## References

### Method

- Smith, R., Self, M., and Cheeseman, P. "Estimating Uncertain Spatial Relationships in
  Robotics". In Autonomous Robot Vehicles, Springer, 1990, pp. 167 to 193.
  DOI: [10.1007/978-1-4613-8997-2_14](https://doi.org/10.1007/978-1-4613-8997-2_14).
  The stochastic map formulation implemented here.
- Thrun, S., Burgard, W., and Fox, D. "Probabilistic Robotics". MIT Press, 2005.
  [https://mitpress.mit.edu/9780262201629/probabilistic-robotics/](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/).
  Chapter 5 for the velocity motion model, chapter 9 for occupancy grid mapping, and
  chapter 10 for EKF-SLAM and maximum likelihood data association.
- Dissanayake, M. W. M. G., Newman, P., Clark, S., Durrant-Whyte, H. F., and Csorba, M.
  "A Solution to the Simultaneous Localization and Map Building (SLAM) Problem". IEEE
  Transactions on Robotics and Automation, 17(3):229 to 241, 2001.
  DOI: [10.1109/70.938381](https://doi.org/10.1109/70.938381). Convergence properties of
  the correlated map.
- Durrant-Whyte, H. and Bailey, T. "Simultaneous Localization and Mapping: Part I". IEEE
  Robotics and Automation Magazine, 13(2):99 to 110, 2006.
  DOI: [10.1109/MRA.2006.1638022](https://doi.org/10.1109/MRA.2006.1638022).
- Bailey, T. and Durrant-Whyte, H. "Simultaneous Localization and Mapping (SLAM): Part
  II". IEEE Robotics and Automation Magazine, 13(3):108 to 117, 2006.
  DOI: [10.1109/MRA.2006.1678144](https://doi.org/10.1109/MRA.2006.1678144).

### Data association

- Neira, J. and Tardos, J. D. "Data Association in Stochastic Mapping Using the Joint
  Compatibility Test". IEEE Transactions on Robotics and Automation, 17(6):890 to 897,
  2001. DOI: [10.1109/70.976019](https://doi.org/10.1109/70.976019). Source of the
  individual compatibility gate used here, and of the joint test that was not
  implemented.

### Consistency and evaluation

- Bar-Shalom, Y., Li, X. R., and Kirubarajan, T. "Estimation with Applications to
  Tracking and Navigation". Wiley, 2001.
  DOI: [10.1002/0471221279](https://doi.org/10.1002/0471221279). Chapter 5 for the NEES
  test and its chi-square confidence bounds.
- Julier, S. J. and Uhlmann, J. K. "A Counter Example to the Theory of Simultaneous
  Localization and Map Building". In IEEE International Conference on Robotics and
  Automation, 2001, pp. 4238 to 4243.
  DOI: [10.1109/ROBOT.2001.933280](https://doi.org/10.1109/ROBOT.2001.933280).
- Bailey, T., Nieto, J., Guivant, J., Stevens, M., and Nebot, E. "Consistency of the
  EKF-SLAM Algorithm". In IEEE/RSJ International Conference on Intelligent Robots and
  Systems, 2006, pp. 3562 to 3568.
  DOI: [10.1109/IROS.2006.281644](https://doi.org/10.1109/IROS.2006.281644).
- Huang, S. and Dissanayake, G. "Convergence and Consistency Analysis for Extended Kalman
  Filter Based SLAM". IEEE Transactions on Robotics, 23(5):1036 to 1049, 2007.
  DOI: [10.1109/TRO.2007.903811](https://doi.org/10.1109/TRO.2007.903811).
- Sturm, J., Engelhard, N., Endres, F., Burgard, W., and Cremers, D. "A Benchmark for the
  Evaluation of RGB-D SLAM Systems". In IEEE/RSJ International Conference on Intelligent
  Robots and Systems, 2012, pp. 573 to 580.
  DOI: [10.1109/IROS.2012.6385773](https://doi.org/10.1109/IROS.2012.6385773). Definition
  of absolute trajectory error.

### Occupancy grid mapping

- Moravec, H. and Elfes, A. "High Resolution Maps from Wide Angle Sonar". In IEEE
  International Conference on Robotics and Automation, 1985, pp. 116 to 121.
  DOI: [10.1109/ROBOT.1985.1087316](https://doi.org/10.1109/ROBOT.1985.1087316). The
  recursive log-odds occupancy update.
- Elfes, A. "Using Occupancy Grids for Mobile Robot Perception and Navigation". Computer,
  22(6):46 to 57, 1989. DOI: [10.1109/2.30720](https://doi.org/10.1109/2.30720).
- Bresenham, J. E. "Algorithm for Computer Control of a Digital Plotter". IBM Systems
  Journal, 4(1):25 to 30, 1965.
  DOI: [10.1147/sj.41.0025](https://doi.org/10.1147/sj.41.0025). The line rasterisation
  used for beam traversal.

### Alternatives considered

- Montemerlo, M. and Thrun, S. "FastSLAM: A Scalable Method for the Simultaneous
  Localization and Mapping Problem in Robotics". Springer Tracts in Advanced Robotics,
  volume 27, Springer, 2007.
  DOI: [10.1007/978-3-540-46402-0](https://doi.org/10.1007/978-3-540-46402-0).
- Kaess, M., Ranganathan, A., and Dellaert, F. "iSAM: Incremental Smoothing and Mapping".
  IEEE Transactions on Robotics, 24(6):1365 to 1378, 2008.
  DOI: [10.1109/TRO.2008.2006706](https://doi.org/10.1109/TRO.2008.2006706).
- Julier, S. J. and Uhlmann, J. K. "Unscented Filtering and Nonlinear Estimation".
  Proceedings of the IEEE, 92(3):401 to 422, 2004.
  DOI: [10.1109/JPROC.2003.823141](https://doi.org/10.1109/JPROC.2003.823141).
- Grisetti, G., Stachniss, C., and Burgard, W. "Improved Techniques for Grid Mapping With
  Rao-Blackwellized Particle Filters". IEEE Transactions on Robotics, 23(1):34 to 46,
  2007. DOI: [10.1109/TRO.2006.889486](https://doi.org/10.1109/TRO.2006.889486).

### Dependencies

| Package | Version | Purpose | Licence |
| --- | --- | --- | --- |
| numpy | >= 2.0 | Array storage, dense linear algebra, random number generation with independent spawned streams | BSD-3-Clause |
| scipy | >= 1.14 | `scipy.stats.chi2` for association gates and NEES confidence bounds, `scipy.ndimage.binary_dilation` for the grid scoring tolerance | BSD-3-Clause |
| matplotlib | >= 3.9 | Trajectory, error, NEES, and occupancy grid figures | Matplotlib licence, a BSD-compatible licence derived from the Python Software Foundation licence |
| pytest | >= 8.3 | Test runner, development only | MIT |
| ruff | >= 0.8 | Linter, development only | MIT |
| mypy | >= 1.13 | Static type checker, development only | MIT |

Citations for the runtime dependencies:

- Harris, C. R. et al. "Array Programming with NumPy". Nature, 585:357 to 362, 2020.
  DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
- Virtanen, P. et al. "SciPy 1.0: Fundamental Algorithms for Scientific Computing in
  Python". Nature Methods, 17:261 to 272, 2020.
  DOI: [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
- Hunter, J. D. "Matplotlib: A 2D Graphics Environment". Computing in Science and
  Engineering, 9(3):90 to 95, 2007.
  DOI: [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

## License

Released under the MIT license. See [LICENSE](LICENSE).
