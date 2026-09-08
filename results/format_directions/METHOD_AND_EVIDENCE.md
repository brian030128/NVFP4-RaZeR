# From isolated tile error to coupled format decisions

## Research question

Can one fixed tile-selection algorithm transfer across models and domains,
without selecting a configuration by its evaluation loss? A method may collect
one common calibration pass per model; the algorithm, objective, candidate
formats and compute budget must remain fixed across models and domains.

## A concrete failure of independent election

Suppose two tile changes each add the same output error vector v, while the
baseline error is -0.6v. Independently, either change lowers squared error from
0.36||v||² to 0.16||v||². Applying both raises it to 1.96||v||². Therefore even
exact isolated tile gains do not justify applying the selected tiles together.
This elementary example motivates testing interactions; it is not a theorem
about downstream language-model loss or a claimed new mathematical result.

## The fixed-candidate objective

For each weight matrix, construct Q0 using canonical FourOverSix and Q1 using
E0M3 with alpha1. Freeze all candidate values, tensor scales and block scales.
Let s select one format for each 8x64 tile and write

    Q(s) = Q0 + sum_t s_t D_t,
    E(s) = Q(s) - W.

D_t is zero outside tile t. Each element remains four bits under either
candidate. The empirical input factor is H=E[xq xq^T], collected once from
FourOverSix-quantized teacher inputs. Ordinary interacting reconstruction uses

    L(s) = tr(E(s) H E(s)^T).

For a tile flip D, its exact change in this objective is

    Delta L = 2 <E H, D> + tr(D H D^T).

Starting from FourOverSix, accept only negative changes, update the residual,
and visit tiles in deterministic K order. Distinct 8-row groups are independent
under this objective. A maximum of eight passes is fixed before testing.

This guarantees nonincrease of the fit objective, up to the stated numerical
tolerance. If a full pass makes no changes, the result is optimal with respect
to a single tile flip. Neither global optimality nor held-out improvement is
guaranteed. Any matrix reaching the pass cap is explicitly reported.

## Why output sensitivity is a separate hypothesis

Reconstruction assigns equal importance to output errors even when downstream
predictions are sensitive to different output directions. The fifth study
approximates teacher-output sensitivity with8x8 output Fisher blocks G_b:

    L_F(s) = sum_b tr(G_b E_b(s) H E_b(s)^T).

Teacher labels are sampled from the BF16 predictive distribution, followed by
one shared backward pass per calibration example. Ground-truth labels and
evaluation losses do not enter the tile decisions. The same update identity
holds with E H replaced by G E H and D H by G D H.

The ablations use G=I and diagonal G. Thus the experiment distinguishes input
geometry, channel sensitivity and within-tile output correlations while keeping
candidates and calibration examples identical. It does not elect the best
ablation separately for each model or domain.

This is a block-diagonal KFAC approximation. It omits input/gradient dependence,
cross-layer interactions, cross-output-block curvature and quantized upstream
trajectories. It is not the exact output KL. Existing work already uses
[Fisher-weighted reconstruction](https://arxiv.org/abs/2102.05426) and
[coordinate descent for quantization](https://arxiv.org/abs/2309.01885).

## Checking the compensation backend

The initial conditional-compensation study used a group update that left no
compensation inside its 64-column tiles. The sixth study therefore runs two
complete columnwise GPTQ paths inside each tile, with one format per path.
The cost-based policy chooses the path with lower accumulated squared
normalized innovations, grouped over eight output rows. Its selected updates
then propagate to future tiles. Dynamic-MSE and E2M1-only controls use exactly
the same columnwise backend.

Tests verify that the accumulated cost equals the final regularized quadratic
error, and that isotropic inputs recover ordinary canonical format election.
This tests a specific backend limitation. It does not erase the failed group
compensation study or establish transfer without the held-out panel.

## Evidence required

Read [REPORT.md](REPORT.md) for every frozen screen and paired contrast, and
[mechanism_summary.json](mechanism_summary.json) for convergence and fit gains.
A lower fit objective alone is insufficient. A larger confirmation study is
justified only by passing the declared cross-model/domain screen, after which
the method must remain locked for new models, domains and generation tasks.

All experiments here emulate quantization on H100. The tile layout is motivated
by an MMA operand shape, not a production-kernel demonstration. External
[SM120 E0M3 experiments](https://github.com/revollllt/sm120-e0m3-mma)
use an undocumented machine-code mode; they do not establish a supported CUDA
interface or the performance of arbitrary per-tile format maps in this project.
