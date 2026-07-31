# Design notes for Differential Drive Slam

## Method selection

### The estimator

The implemented estimator is EKF-SLAM in the stochastic map formulation of Smith,
Self, and Cheeseman (1990), following the presentation in chapter 10 of Thrun,
Burgard, and Fox, Probabilistic Robotics (2005). One Gaussian is maintained over the
joint state `[x, y, theta, m1x, m1y, ...]`, with a dense covariance that holds every
robot to landmark and landmark to landmark correlation.

The formulation was chosen for the problem stated in the README because that problem
is small, feature-based, and specified in the terms EKF-SLAM is written in. The world
is planar, the landmarks are point features observed in range and bearing, the motion
model is a two-parameter body twist, and the map holds tens of landmarks rather than
tens of thousands. In that regime the extended Kalman filter is close to optimal, and
the questions worth asking of the code, whether the correlations are maintained
correctly, whether the reported covariance matches the error, and how the cost grows
with map size, are all directly visible in the state and covariance rather than
mediated by a solver or a particle set.

The assumptions the method depends on, and where they hold here:

- **The posterior is approximately Gaussian.** This holds when the linearisation error
  over one time step is small relative to the covariance. With a step of 0.1 s, a speed
  of 1.0 m/s, and a heading standard deviation that stays near 0.010 rad, the arc
  travelled per step is 0.1 m and the heading change is at most 0.079 rad, so the
  second-order terms of both the motion and the measurement model are far below the
  noise. It fails when the heading uncertainty becomes large, and that is exactly the
  regime where EKF-SLAM is documented to become inconsistent (Julier and Uhlmann 2001;
  Bailey et al. 2006; Huang and Dissanayake 2007).
- **The landmarks are static.** The state has no velocity for landmarks and the process
  noise adds nothing to the map block, so a landmark that moves will be fought over by
  successive measurements and will drag the pose estimate with it.
- **Measurement noise is zero mean, Gaussian, and independent between detections.** The
  simulator generates exactly this, so the accuracy figures in the README measure the
  estimator rather than a model mismatch. Real range and bearing detectors have
  range-dependent bias and correlated bearing error, which this repository does not
  model.
- **At most one measurement per landmark per time step.** The maximum likelihood policy
  assigns each measurement independently, so nothing prevents two measurements in the
  same batch from being matched to the same landmark. In this simulator that cannot
  happen because each landmark generates at most one detection, but on a real detector
  with split returns it can.

### Prediction in block form

The textbook derivation writes prediction with an identity-padded Jacobian
`G = I + F^T (Gx - I) F`, which is an operation on the full `(3 + 2N)` square matrix.
The implementation instead updates the robot block and the robot to map cross blocks
directly. The two are algebraically identical because the map block of the padded
Jacobian is the identity and the process noise is zero outside the robot block. The
block form was chosen because it makes the structure explicit: prediction touches
`O(N)` entries, not `O(N^2)`, and only correction is quadratic.

### Joseph form correction

The covariance update uses `(I - K H) P (I - K H)^T + K Q K^T` rather than
`(I - K H) P`. The Joseph form costs one extra dense matrix product per measurement.
It was chosen because it is symmetric by construction and remains positive
semidefinite for any gain, not only the optimal one, so accumulated rounding cannot
push the covariance indefinite over a run of several thousand corrections. The
covariance is additionally symmetrised after every prediction and correction, and the
invariant is asserted over a full run in the test suite.

### Data association

Association is individual compatibility gating followed by maximum likelihood
selection, in the sense of Neira and Tardos (2001) for the gate and Probabilistic
Robotics table 10.1 for the selection.

Two details differ from the most common textbook presentation. First, the selection
ranks candidates by `nu^T S^-1 nu + log det(2 pi S)` rather than by the Mahalanobis
distance alone. Candidate landmarks generally have different innovation covariances,
in particular a freshly initialised landmark has a much larger one than a landmark
seen for fifty steps, and ranking by distance alone is therefore not the maximum
likelihood rule. Second, there are two thresholds rather than one: a measurement
inside the 0.99 chi-square quantile of the nearest landmark is matched, a measurement
outside the 0.9999 quantile of every landmark initialises a new landmark, and a
measurement between the two is discarded. Discarding is the conservative choice.
Accepting an ambiguous measurement risks a false association, which is unrecoverable
in a filter with no delete operation; initialising one risks a duplicate landmark,
which wastes state and can later attract the correct measurements away from the real
landmark. On the default scenario the rule discards 0.93 percent of detections and makes
no incorrect assignment across 20407 detections.

### Occupancy grid mapping

The grid uses the recursive log-odds update of Moravec and Elfes (1985), with the
inverse sensor model reduced to two constants: a fixed decrement for every cell a beam
passes through and a fixed increment for the cell it terminates in. The accumulated
value is clamped to a configurable interval.

The clamping is not cosmetic. Without it, a cell observed a thousand times reaches a
log odds of several hundred, and no amount of contrary evidence can move it back within
the lifetime of the run. Clamping bounds the number of contrary observations needed to
flip a cell, which is the standard remedy and the reason the bound is exposed as a
parameter rather than hard-coded.

Beam traversal uses the integer digital differential analyser form of Bresenham's line
algorithm (Bresenham 1965), which produces the same eight-connected cell chain as the
classical incremental formulation for integer endpoints while remaining a vectorised
NumPy expression.

### Evaluation

Absolute trajectory error is computed directly, with no alignment step. The
Umeyama-style alignment used for real datasets (Sturm et al. 2012) exists because the
estimated and ground truth trajectories live in different frames; here the simulator
fixes the world frame and the filter is initialised in it, so aligning would remove a
genuine part of the error.

Landmark error is scored against the ground truth landmark that generated the detection
which initialised each slot. That mapping is recorded by the simulator and never shown
to the filter. Nearest-neighbour matching between the estimated and true maps was
rejected: it flatters a map whose landmarks have drifted towards each other and it
cannot distinguish a duplicate landmark from a correct one.

Consistency uses the NEES of the three-dimensional pose error against the reported
marginal covariance, with the chi-square bounds of Bar-Shalom, Li, and Kirubarajan
(2001), chapter 5. The primary test is the per-time-step ensemble average over
independent Monte Carlo runs, not the time average of a single run, because consecutive
NEES samples within a run are strongly correlated and treating them as independent
produces an interval far tighter than the evidence supports. Both intervals are printed,
and the reports state which one the verdict rests on.

## Rejected alternatives

### FastSLAM and Rao-Blackwellised particle filters

Montemerlo and Thrun's FastSLAM factors the posterior into a particle set over
trajectories with an independent small filter per landmark per particle. It would have
bought two things: a cost that grows with the log of the landmark count rather than the
square, and the ability to represent a multi-modal pose posterior, which matters when
data association is genuinely ambiguous.

It would have cost the property that makes this repository useful as a reference: the
joint covariance would no longer exist as an object, so the correlation structure that
the whole exercise is about could not be inspected, and the NEES consistency test would
have to be replaced by a particle-based diagnostic with its own pitfalls, including
particle depletion masquerading as confidence. The scenario here has tens of landmarks
and unambiguous association, so neither benefit would have been exercised. Grisetti,
Stachniss, and Burgard (2007) is the reference to follow if the particle formulation is
wanted later.

### Graph-based smoothing

Incremental smoothing and mapping, in the iSAM form of Kaess, Ranganathan, and Dellaert
(2008), solves for the whole trajectory rather than the latest pose. It is what a
production system would use today. It would have bought relinearisation of past states,
which removes the main source of EKF inconsistency, and sparsity, since the information
matrix of the full problem is sparse where the covariance of the filtered problem is
dense.

It was rejected because it changes the subject. A smoother is a different estimator
answering a different question, its cost is dominated by the sparse factorisation rather
than by the model, and implementing one well means implementing an incremental QR
update, a variable ordering heuristic, and a nonlinear solver. That is a separate
repository, and the honest comparison is that this one implements the filter that the
smoother replaced.

### Unscented Kalman filter

Replacing the analytic Jacobians with the sigma point transform of Julier and Uhlmann
(2004) would have removed the linearisation error without changing the state
representation, and it is a small change to make.

It was rejected because the sigma point set has to be regenerated over the full
`(3 + 2N)` state, so the cost per step rises with the map size, and because the analytic
Jacobians are themselves part of what this repository is meant to show: they are
derived, documented, and checked against central finite differences in the test suite.
The linearisation error at the step size used here is far below the noise, so the
accuracy gained would not be measurable.

### Joint compatibility branch and bound

Neira and Tardos (2001) test the joint compatibility of a whole set of associations
rather than each in isolation, which resolves the ambiguities that the individual gate
must discard. It would have bought robustness in dense maps where several landmarks
fall inside one another's gates.

It was rejected because it is exponential in the batch size without careful pruning,
and because on this scenario there is nothing to gain: the individual gate already
achieves a perfect association record, and the 0.93 percent of measurements it discards
would at best be recovered, not corrected. It becomes the right choice when the landmark
density rises relative to the sensor resolution, which is the condition under which this
repository's association policy degrades.

### Landmark deletion and map management

A quality counter per landmark, deleting landmarks that are observed less often than
expected, would have removed the roughly one spurious landmark per run that the maximum
likelihood policy creates. It was rejected as scope: it introduces a state removal
operation whose interaction with the covariance and with the recorded slot to identity
mapping needs its own tests, and the spurious landmarks are counted and reported rather
than hidden.

### Scan matching for the occupancy grid

Aligning consecutive laser scans, rather than mapping at the filtered pose, would have
made the grid sharper by removing the pose error from the map. It was rejected because
it would make the grid a second, independent localisation system, so the grid would no
longer measure what it is here to measure, which is what a map built on top of this
filter's pose estimate actually looks like. The tolerance sweep in the README exists to
make that error visible rather than to hide it.

## Known limitations

### EKF-SLAM scales quadratically in the landmark count

This is the binding constraint on the implementation.

The covariance is a dense `(3 + 2N)` by `(3 + 2N)` matrix, so memory grows as `O(N^2)`.
Each measurement correction forms `H P`, the Kalman gain, and the Joseph form product,
all of which are `O(N^2)` per measurement, so a time step with `k` visible landmarks
costs `O(k N^2)`. Data association adds another factor: scoring one measurement against
every landmark evaluates `N` innovations, each `O(N^2)` because of the `H P H^T`
product, giving `O(k N^3)` per step in the association path as implemented. Prediction,
by contrast, is only `O(N)`, since it touches the robot block and the robot to map cross
blocks alone.

Where that becomes binding, with the numbers from this repository: the default scenario
has 20 landmarks, a 43 by 43 covariance, roughly 6.3 measurements per step, and completes
640 steps in about 2 seconds. The quadratic term is invisible at that size. It becomes
the limit at a few hundred landmarks, where the covariance passes several megabytes and
a single step costs more than the 0.1 s the step represents, so the filter can no longer
run in real time on the trajectory it is estimating. At a few thousand landmarks the
dense covariance alone is hundreds of megabytes and the approach is no longer viable at
all. Concretely, a 10 m by 10 m room with 20 landmarks is comfortable; a building floor
with a thousand is not. There is no tuning that removes this, because the density of the
covariance is a property of the formulation: every landmark becomes correlated with
every other one as soon as the robot observes both.

The remedies all mean changing the estimator rather than optimising it. Submapping bounds
`N` per filter and stitches the submaps afterwards. The information filter form, in which
the matrix is sparse to a good approximation, trades the covariance for an information
matrix and needs a solve to recover any marginal. FastSLAM factors the map. Graph-based
smoothing keeps the sparsity exactly and pays for it with a factorisation.

### The consistency verdict is configuration-specific

The Monte Carlo study reports an ensemble average NEES of 2.91 against an expected 3,
with 97.5 percent of time steps inside the per-step 95 percent interval against a nominal
95 percent. That is consistent, marginally on the conservative side. This is not a
general property of EKF-SLAM, and it should not be read as a refutation of the
inconsistency results in the literature. Three features of the default scenario push the
verdict in the conservative direction: the landmarks are dense and
observed from every heading, so the pose is over-determined at almost every step; the
heading uncertainty stays near 0.010 rad, which is the regime where the linearisation is
accurate; and the association policy discards its 0.93 percent of ambiguous measurements,
throwing away information and thereby inflating the reported covariance relative to the
error. Reducing the landmark density, increasing the bearing noise, or lengthening the
interval between loop closures should move the verdict towards optimistic, which is the
direction that Julier and Uhlmann (2001) and Bailey et al. (2006) predict and the
direction that matters for safety.

### Landmarks are never removed

There is no delete operation. A landmark initialised from a spurious detection stays in
the state for the rest of the run, costs a row and a column in the covariance forever,
and can attract correct measurements away from the real landmark it duplicates. The
maximum likelihood runs end with 20.4 landmarks on average against a true count of 20.
Removing this needs a per-landmark quality statistic, a removal rule, and a covariance
row and column deletion, together with the tests that show the remaining state is still a
valid Gaussian.

### The heading is treated as a real number

Angles are wrapped into `[-pi, pi)` and bearing innovations are wrapped before use, which
is correct as long as the true error is well inside half a turn. It is not a proper
treatment of the circular topology of the heading. A pose with a heading standard
deviation approaching 1 rad has a covariance that no longer describes the distribution on
the circle, and the Gaussian mean itself becomes the wrong summary. Fixing this means
estimating on SE(2) with an error-state formulation rather than treating the heading as a
real number.

### The occupancy grid assumes an ideal beam

The inverse sensor model has no beam width, no incidence angle dependence, and no maximum
range mixture component. A real laser return at a grazing incidence is both less likely
and less accurate than a normal-incidence return, and a real sensor produces a fraction
of returns at maximum range that carry no free space information. Adding these means
replacing the two constants with a function of range and incidence and re-deriving the
cell update. The grid also has no notion of the robot footprint, so cells the robot
itself occupies are marked free.

### The simulated world is convex and static

The environment is axis-aligned wall segments and static point landmarks. There are no
moving obstacles, no featureless stretches where the landmark detector returns nothing
for an extended period, and no perceptual aliasing, that is, no two places that look the
same. All three are the failure modes that separate a working SLAM system from a
demonstration, and none of them is exercised here. In particular, the association figures
in the README should be read as an upper bound on what the policy achieves, not as an
estimate of its performance on real data.
