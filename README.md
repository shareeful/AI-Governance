# AI Governance Pipeline Framework

Reference implementation of **"An AI Governance Pipeline Framework with Trustworthy AI Practice
for High-Risk AI Systems"** (Hassan & Islam).

The framework is a **runtime governance control plane** that sits around the point of inference of
a high-risk AI system. It compiles five obligations derived from the EU AI Act into per-execution
checks, enforces them through coordinated input, reasoning, and output guards, and adds a
reference-free **Model Behavioural Attestation** layer that monitors the deployed model against its
certified behavioural baseline. Every admitted or withheld execution is written to a signed,
hash-chained audit log.

This repository contains the framework, the guard instantiations, the calibration and fingerprint
procedures, the attack constructions, the published baselines, and the scripts that reproduce every
table and figure of the experimental evaluation.

---

## Table of contents

- [What is implemented](#what-is-implemented)
- [Design commitments](#design-commitments)
- [Installation](#installation)
- [Data you must supply](#data-you-must-supply)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [The four-phase lifecycle](#the-four-phase-lifecycle)
- [Command reference](#command-reference)
- [Reproducing the evaluation](#reproducing-the-evaluation)
- [Governing a new system](#governing-a-new-system)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Scope of the guarantees](#scope-of-the-guarantees)
- [Citation](#citation)
- [Licence](#licence)

---

## What is implemented

### The policy model — five runtime-verifiable clauses

| Clause | Measure | Threshold | Enforcing guard | Regulatory basis | Guarantee |
|---|---|---|---|---|---|
| Non-discrimination | Counterfactual invariance `μ_nd` | `θ = 1` | Output | Art. 10 | Exact |
| Data minimisation | Absence of any item beyond the minimum set `μ_dm` | `θ = 1` | Input / Reasoning / Output | Art. 10(5); GDPR Art. 5(1)(c) | Exact |
| Reasoning faithfulness | Causal ablation `μ_rf` | calibrated | Reasoning | Art. 13 | `≤ α` at confidence `1 − δ` |
| Precondition compliance | Proportion of required checks recorded before commit `μ_pc` | `θ = 1` | Reasoning | Art. 12 | Exact |
| Factual groundedness | Entailment of claims against evidence `μ_fg` | calibrated | Output | Art. 15 | `≤ α` at confidence `1 − δ` |

Every measure is oriented so that a higher value indicates greater compliance. The three exact
clauses are decided deterministically; the two learned clauses are calibrated by Learn-then-Test so
that the probability of certifying a violating execution is bounded in finite samples.

Clause records are compiled from configuration (`policy.clauses`) into an executable schema that
names the clause, its regulatory basis, its measure and threshold, its checker, the execution
components the checker may read, and the action taken on failure. Read access is enforced: a clause
that declares access to a component it may not read is refused at binding time.

### The architecture

- **Input guard** — injection and jailbreak classification with an LTT-calibrated decision point,
  sanitisation where a request can be salvaged, permitted-action scope enforcement, evidence
  provenance admission, and de-identification down to the minimum-necessary set. The protected
  attribute is retained under Art. 10(5) in a **segregated store** readable only by the
  counterfactual checker and excluded from the trace, the response, and the log.
- **Reasoning guard** — causal ablation of each stated reason with re-execution of the decision,
  presence of every required check before the commit point, and data minimisation over the trace.
- **Output guard** — counterfactual re-execution across every value of each protected attribute,
  natural-language-inference entailment of each claim against the supplied evidence, content-safety
  screening, and data minimisation over the response.
- **Human oversight** — risk-band routing between human-in-the-loop and human-on-the-loop. A
  high-band execution is *held* for approval, which is distinct from a guard rejection. Reviewers
  are real: the queueing reviewer writes the certified trace to a review queue and returns
  `pending`; the recorded-decision reviewer replays approvals a human has actually logged. Nothing
  approves an execution on a reviewer's behalf.
- **Model Behavioural Attestation** — four reference-free signals (faithfulness distribution,
  feature-reliance vector, STRIP perturbation entropy, paraphrase agreement) summarised into a
  signed fingerprint at certification; at runtime each signal is compared against its certified
  distribution over a sliding window (two-sample Kolmogorov–Smirnov for the scalars,
  Jensen–Shannon divergence for the reliance vector), converted to conformal p-values against the
  certification workload, combined by a Holm correction, and reduced to `D_t = −log p_min^Holm`.
  The verdict is *attested* exactly when `D_t ≤ τ`, with `τ` calibrated so the false-alarm rate on
  an unmanipulated model is at most `β`.
- **Trust layer** — the execution-level risk score
  `R(e) = w_att·1[D_t > τ] + Σ_c w_c·(1 − μ_c(e))` with weights that must sum to one, the
  overrides that force the high band, an append-only Ed25519-signed and SHA-256 hash-chained
  ledger, continuous policy monitoring, and drift-triggered re-certification from screened runtime
  evidence.

### Evaluation machinery

- **Seven governed systems** — Retinal-OCT, Fraunhofer-Beads, CheXpert, HAM10000 (imaging);
  MIMIC-IV, German Credit, COMPAS (tabular).
- **External benchmarks** — BBQ, i2b2/UTHealth, the early-answering / added-mistake /
  biased-context faithfulness protocols, FEVER, HaluEval, AdvBench, JailbreakBench. Precondition
  compliance has no public per-execution benchmark and uses an internal required-check deletion;
  every row it produces is flagged `external_benchmark: false`.
- **Clause-enforcement baselines** — ungoverned, input-only, output-only, trace-aware,
  action-level, reference-based, and the proposed plane.
- **Model-integrity baselines** — STRIP, Neural Cleanse, activation clustering, spectral
  signatures, MNTD, and an MMD drift detector, each reporting its own access assumptions.
- **Attack families** — A1 BadNets trigger backdoor, A2 label-flip poisoning, A3 silent model
  substitution (shifted fine-tune or post-training quantisation), A4 attestation-aware adaptive
  adversary that optimises a backdoor objective subject to holding each monitored signal inside its
  calibrated bound.
- **Statistical protocol** — repeats over the configured seeds, 95 % bias-corrected and
  accelerated bootstrap intervals, two-sided Wilcoxon signed-rank tests paired by system and seed,
  and Holm–Bonferroni correction within each table.

---

## Design commitments

These are properties of the code, not aspirations.

**No synthetic, dummy, or placeholder data.** Nothing in this repository fabricates a dataset, a
benchmark case, a model checkpoint, or a result. Every dataset and benchmark path is `null` in the
shipped configuration and must be set to a real resource. A missing or unreadable path raises
`MissingResourceError` with an explanation rather than falling back to generated data.

**No hard-coded values.** Thresholds, risk targets, weights, seeds, window sizes, model
identifiers, class names, protected-attribute domains, required-check lists, augmentation
parameters, partition tolerances, baseline hyperparameters, and dataset layouts all come from
configuration. The two learned thresholds cannot be set in configuration at all — attempting to do
so is refused, because they are the output of Learn-then-Test in Phase 2.

The only literals left in the source are mathematical or format constants, not tunable parameters:
the 8-bit pixel scale (taken from `np.iinfo(np.uint8).max`), the MAD-to-sigma constant `1.4826` in
Neural Cleanse, the BCa acceleration coefficients, the SHA-256 hex digest length, the seconds-to-
milliseconds factor, numerical floors, and the 5th/95th fingerprint percentiles that the
specification itself fixes.

**No datasets in the repository.** The six public systems and every external benchmark are obtained
from their original sources under their own terms. The Fraunhofer-Beads pilot data is proprietary
and is not redistributable.

**Fail-closed.** Every guard withholds when it cannot certify. A checker that has not been
calibrated refuses to issue a verdict rather than using a default cut-off.

**No comments in the source.** All explanation lives in this README, by request.

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/mhassan720/ai-governance-pipeline.git
cd ai-governance-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The Fraunhofer-Beads pilot additionally needs the detection extra:

```bash
pip install -e ".[detection]"
```

---

## Data you must supply

Nothing below ships with the repository. Obtain each resource from its original source under its
own licence and record the path in your local configuration.

### Governed systems

| System | Source | Notes |
|---|---|---|
| Retinal-OCT | Kermany et al., published repository | Four classes: CNV, DME, DRUSEN, NORMAL |
| CheXpert | Irvin et al., Stanford ML Group | Requires the Stanford data-use agreement |
| HAM10000 | Tschandl et al., Harvard Dataverse | |
| MIMIC-IV | Johnson et al., PhysioNet | Requires PhysioNet credentialed access |
| German Credit | Hofmann, UCI ML Repository | |
| COMPAS | Angwin et al., ProPublica | Used to measure the fairness checks, not as an endorsement |
| Fraunhofer-Beads | Fraunhofer IBMT, under project agreement | Not redistributable; direct requests to Fraunhofer IBMT |

### External benchmarks

| Clause / check | Benchmark | Expected format |
|---|---|---|
| Non-discrimination | BBQ | JSONL, fields configurable under `benchmarks.non_discrimination` |
| Data minimisation | i2b2/UTHealth de-identification corpus | Paired `.txt` / `.xml` documents beneath one root |
| Reasoning faithfulness | Early-answering, added-mistake, biased-context protocols | Three JSONL files, each carrying an answer and its perturbed answer |
| Factual groundedness | FEVER and HaluEval | JSONL |
| Injection / jailbreak | AdvBench and JailbreakBench | CSV and JSONL respectively |
| Content safety | Your organisation's moderation corpus | JSONL with a text field and an unsafe label |

Field names are configurable throughout, so a differently-named export of the same corpus needs a
configuration change rather than a code change.

### Guard component checkpoints

Four checker models must be supplied as local paths or resolvable model identifiers, with the
label name each exposes:

- an injection / jailbreak sequence classifier (`checkers.injection`),
- a de-identification token classifier (`checkers.deidentification`),
- a natural-language-inference model (`checkers.entailment`),
- a content-safety classifier (`checkers.safety`).

Each is loaded through the transformers auto-classes. If a checkpoint cannot be resolved, loading
fails with `MissingResourceError`; an untrained model is never substituted.

### Split discipline

Each dataset must be partitioned into **disjoint** training, calibration, and test splits, plus a
`drift` split drawn from a genuinely shifted source (a different acquisition site, device, or
period) for RQ4. The calibration split alone sets the learned thresholds and the fingerprint; the
test split alone reports the metrics.

### Augmentation

`systems.<name>.augmentation` implements the leakage-safe scheme of the paper: `views_per_image`
views per training image, built from rotation and flips, brightness and contrast jitter, additive
noise, motion blur, and synthetic occlusions. It is enabled for the Fraunhofer-Beads pilot, where
240 original images at 16 views each give the 3,840 training views the paper reports, and disabled
for the larger systems that do not need it.

Augmentation touches the **training split only**. Before writing anything it checks that no
training identifier also appears in a held-out split and refuses outright if one does, so no
augmented view of a validation or test image can reach calibration. `retain_originals: false`
means the count is exactly `views_per_image` per image; set it to `true` to keep the originals in
training as well. Output is written as PNG, and a lossy format is refused so that augmentation
artefacts are not compounded by compression artefacts.

For the imaging systems, set `dataset.enforce_group_disjoint_splits: true` and an appropriate
`dataset.identifier_from` strategy. With `patient_field`, identifiers are parsed from the
`<class>-<patient>-<scan>` filename convention and the loader **refuses to proceed** if the same
patient appears in more than one split. This matters: a naive image-level random split of the
Kermany OCT corpus leaks patients across the train/test boundary and inflates every reported
number. Repartition at the patient level before certifying.

---

## Configuration

Configuration is layered. Later files override earlier ones, key by key:

```bash
agp --config configs/default.yaml \
    --config configs/systems/retinal_oct.yaml \
    --config configs/local.yaml \
    <command>
```

- `configs/default.yaml` — the framework defaults: clause records, calibration targets, attestation
  parameters, risk weights, baselines, attacks, and the experimental protocol. Every path and every
  site-specific value is `null`.
- `configs/systems/*.yaml` — one file per governed system, carrying its kind, model specification,
  attribution grid, protected attributes, action register, required-check lists, and rationale
  templates.
- `configs/local.yaml` — **yours**, and git-ignored. Copy `configs/example.local.yaml` and fill in
  the paths. This is the only file you need to edit to run the framework on your data.

### Keys you must set

| Key | Meaning |
|---|---|
| `paths.artifact_root` | Where checkpoints, fingerprints, ledgers, results, tables, and figures are written |
| `trust.ledger.private_key` / `public_key` | Ed25519 key pair for the audit log; generate with `agp keygen` |
| `oversight.reviewer_identifier` | The identifier recorded against reviewed decisions |
| `checkers.*.model` and the matching `*_label` | The four guard component checkpoints |
| `benchmarks.*` | The external benchmark paths |
| `systems.<name>.dataset.*` | Dataset root, split layout, and the protected-attribute metadata table |
| `systems.<name>.protected_attribute_domains` | The finite domain of each protected attribute, from your data |
| `systems.<name>.action_register` | Permitted actions, their high-risk flag, and their required-check lists |
| `experiments.systems` | Which systems the experiments run over |

### Keys you may tune

`calibration.alpha` (0.05) and `calibration.delta` (0.10) set the learned-clause risk targets.
`attestation.calibration.alpha` (0.01) is the detector false-alarm budget `β`, and
`attestation.window` (200) is the sliding window `w`. `trust.risk.weights` encodes your deployment's
impact assessment and is validated to sum to one. `experiments.seeds` defaults to the ten seeds
used in the paper.

The detector emits one statistic per **full** window, so any stream it scores must be longer than
`attestation.window`. `experiments.minimum_detector_windows` (50) sets how many sliding windows an
experiment requires, and a split too short to supply them is refused by name rather than silently
producing a statistic or two. If your test split is small, shorten the window or lower this value.

### Why some thresholds cannot be configured

Setting `policy.clauses.reasoning_faithfulness.threshold` or
`policy.clauses.factual_groundedness.threshold` to a number is **refused**. Those thresholds are
produced by Learn-then-Test against your calibration split, and hard-coding them would void the
finite-sample risk bound that the corresponding proposition claims. Likewise, the three exact
clauses are refused any threshold other than `1.0`.

---

## Quick start

```bash
cp configs/example.local.yaml configs/local.yaml
# edit configs/local.yaml so every path points at your real data

CFG="--config configs/default.yaml --config configs/systems/retinal_oct.yaml --config configs/local.yaml"

agp $CFG keygen
agp $CFG compile
agp $CFG train    --system retinal_oct --seed 42
agp $CFG certify  --system retinal_oct --seed 42
agp $CFG run      --system retinal_oct --seed 42 --split test --limit 200
agp $CFG verify-ledger /path/to/artifacts/ledger/retinal_oct/seed_42.jsonl
```

---

## The four-phase lifecycle

**Phase 1 — compile and bind.** `agp compile` turns the five clauses into executable records.
Binding attaches each record to a system through its execution interface, verifies that the system
supports controlled re-execution (required by the counterfactual and ablation runners), checks that
every protected attribute has a non-empty domain, and rejects an action whose required-check list
is empty, since the precondition clause would otherwise certify vacuously.

**Phase 2 — certify.** `agp certify` calibrates the injection, entailment, and safety decision
points on their benchmarks; calibrates the two learned clause thresholds by Learn-then-Test against
the calibration split; records the behavioural fingerprint over a held-out workload; builds the
conformal null distribution by resampling windows from the certification workload; and calibrates
the detector threshold `τ`. The fingerprint is Ed25519-signed, and a calibration report is written
to `artifacts/certification/<system>/seed_<n>.json`.

If no candidate threshold controls the risk at the configured `(α, δ)`,
`NoAdmissibleThresholdError` is raised and reports the most permissive attainable empirical risk.
That is the correct outcome: the deployment cannot be certified at that target on that calibration
set, and the answer is more calibration data or a weaker target, not a hand-set threshold.

**Phase 3 — enforce.** `agp run` executes Algorithm 3 per request: input guard, inference,
reasoning guard, output guard, attestation, risk scoring, and oversight routing. A guard rejection
withholds immediately and fails closed. A not-attested verdict withholds independently of the
clause guards. A high-band execution is held for human approval — held, not blocked. Every
execution is sealed into the signed, hash-chained record of Template T3.

**Phase 4 — govern.** The monitor tracks the reject rate, the not-attested rate, and the shift of
the input distribution away from the calibration set. When any exceeds its configured bound,
re-certification re-enters Phase 2. Candidate data are drawn from the signed runtime records and
screened first: withheld executions, not-attested executions, and anything a contamination hook
rejects are excluded, and re-certification proceeds only from the approved subset.
`agp verify-ledger` re-walks the chain and verifies every signature.

---

## Command reference

| Command | Purpose |
|---|---|
| `agp keygen` | Generate the Ed25519 ledger key pair |
| `agp compile` | Compile the five clauses and print the records (Phase 1) |
| `agp train` | Train a governed decision model |
| `agp shadow-pool` | Train the clean and trojaned shadow models the MNTD baseline requires |
| `agp certify` | Calibrate thresholds and record the fingerprint (Phase 2) |
| `agp run` | Enforce the certified policy over a split (Phase 3) |
| `agp experiment [names...]` | Run `rq1`, `rq2`, `rq2_sensitivity`, `rq3`, `rq4`, `rq5`, `ablation`, or all |
| `agp report` | Render every table and figure from the stored results |
| `agp verify-ledger <path>` | Verify an audit log's hash chain and signatures (Phase 4) |

`--system` and `--seed` are repeatable and default to `experiments.systems` and
`experiments.seeds`.

Object detectors are trained with their own framework; point
`systems.<name>.model.weights` at the resulting checkpoint. `agp train` says so rather than
pretending to train one.

---

## Reproducing the evaluation

```bash
CFG="--config configs/default.yaml"
for s in configs/systems/*.yaml; do CFG="$CFG --config $s"; done
CFG="$CFG --config configs/local.yaml"

agp $CFG keygen
agp $CFG train
agp $CFG shadow-pool --system retinal_oct
agp $CFG experiment
agp $CFG report
```

or equivalently `scripts/reproduce.sh`.

Outputs land under `paths.artifact_root`:

```
results/    one JSON per experiment, with per-seed rows, bootstrap intervals, and paired tests
tables/     CSV and LaTeX for every reported table
figures/    the clause heat-map, baseline bars, ROC curves, sensitivity sweeps,
            calibration curves, and drift curves
```

| Experiment | Question | Principal outputs |
|---|---|---|
| `rq1` | Does the plane block violations of every clause across every system, and at what cost to compliant traffic? | Clause-by-system recall heat-map, modality table, baseline comparison |
| `rq2` | Does reference-free attestation catch a model whose output remains reference-consistent? | AUC against six published detectors over A1–A4, plus the leave-one-signal-out and reference-based controls |
| `rq2_sensitivity` | Does detection depend on a strong attack? | Detection at the `β` budget across the poison-rate and trigger-fraction sweeps |
| `rq3` | Do the Learn-then-Test thresholds deliver the guarantee they promise? | Empirical certified-violation rate against target `α` over `[0.01, 0.10]` |
| `rq4` | Do the guarantees survive drift? | Violation rate over the operating window, static against drift-triggered re-certification |
| `rq5` | What does governance cost? | Per-component added latency and retained throughput |
| `ablation` | Is any layer redundant? | Each layer removed in turn, measured against the metric it governs |

The ablation assesses the oversight layer on the missed-referral rate rather than on detection
recall, because that is the only one of the four metrics it can move.

Every number in `results/` is computed from your data. The repository ships no reference values,
and no experiment will run without the real resources it needs.

---

## Governing a new system

The plane asks two things of a system: that it expose its execution interface, and that it permit
controlled re-execution on a modified input or trace.

1. Subclass `agp.systems.base.InstrumentedSystem` and implement `predict_action`,
   `predict_action_with_groups_removed`, `attribution`, `decision_scores`, `substitute_attribute`,
   `perturbations`, `paraphrases`, and `recorded_checks`. The base class derives the reasoning
   trace, the evidence, and the response from your attributions and rationale templates.
2. Add a configuration file under `configs/systems/` declaring the model, the feature groups, the
   protected attributes and their domains, the action register with its required-check lists, and
   the rationale templates.
3. Register the kind in `agp.systems.registry.build_system`.
4. Certify it in Phase 2. No guard changes.

For systems whose predictions carry no decomposable trace, reasons are derived from saliency or
attribution regions and a reason is ablated by occluding its region — this is how the imaging
systems work here, and it preserves the soundness property of the guard semantics.

---

## Repository layout

```
src/agp/
  config.py              layered configuration with fail-loud resolution
  execution.py           the execution tuple (x, r, a, y) and the governed-system interface
  certify.py             Phase 2 assembly: checkers, thresholds, fingerprint, pipeline
  training.py            governed-model training and the MNTD shadow pool
  cli.py                 the agp command-line interface
  policy/                clause records, the five measures, guard semantics, compile and bind
  calibration/           Learn-then-Test, Hoeffding-Bentkus bounds, conformal p-values, Holm
  checkers/              injection, de-identification, scope, provenance, entailment, safety
  guards/                input, reasoning, output guards and human oversight
  attestation/           the four signals, the fingerprint, the streaming detector
  trust/                 risk scoring, the signed hash-chained ledger, monitoring, re-certification
  runtime/               Algorithm 3, the admit/withhold decision
  systems/               the seven governed systems, imaging and tabular, and the
                         leakage-safe augmentation scheme
  benchmarks/            external benchmark loaders and the violation-embedding harness
  baselines/             clause-enforcement baselines and six model-integrity detectors
  attacks/               A1 backdoor, A2 label flip, A3 substitution, A4 adaptive
  stats/                 BCa bootstrap, Wilcoxon, Holm-Bonferroni, clause and detection metrics
  experiments/           RQ1-RQ5 and the ablation
  reporting/             tables and figures
configs/                 default, per-system, and an example local override
tests/                   unit tests for the calibration, policy, measure, attestation, trust,
                         statistics, and configuration layers
scripts/                 end-to-end reproduction
```

---

## Testing

```bash
pytest
```

96 tests cover what can be verified without a dataset:

- **Calibration** — Learn-then-Test returns the most permissive admissible threshold, no more
  permissive candidate would have passed, and it refuses outright when none controls the risk; the
  detector threshold bounds the false-alarm rate; Holm adjustment matches its step-down definition.
- **Policy** — every clause belongs to exactly one decision family; an exact clause refuses any
  threshold but 1.0; a learned clause refuses a configured threshold and refuses to decide before
  calibration.
- **Measures** — each of the five measures at its boundaries, including counterfactual dependence,
  post-hoc reasons, checks recorded after the commit point, and partial entailment.
- **Attestation** — the fingerprint round-trips; the detector statistic rises under a shifted
  stream; `score_stream` yields one statistic per full window and refuses a stream shorter than the
  window; leave-one-signal-out preserves the remainder.
- **Augmentation** — every view stays within the unit interval; motion blur reduces local variance
  and is the identity at unit kernel; occlusion writes the configured fill; the view count is
  exactly `views_per_image`; a lossy output format is refused; and a training image that also
  appears in a held-out split is refused outright, so augmentation cannot leak.
- **Trust** — risk weights must sum to one; each override forces the high band; the ledger chains,
  survives reopening, and detects a tampered record; re-certification screening excludes withheld
  and not-attested evidence.
- **Runtime (Algorithm 3)** — a compliant execution is admitted; an input-guard rejection withholds
  before inference; a missing precondition withholds and refers; an attribute-dependent action and
  an ungrounded response are each caught by the output guard; a not-attested execution is withheld
  and referred; and a high-risk action is **held** for approval rather than blocked, then released
  once a reviewer approves it. The segregated protected-attribute store is discarded on every exit
  path and refuses reads afterwards.
- **Configuration** — layering, fail-loud resolution of absent and null keys, and refusal of a
  missing path rather than substitution.

The runtime tests drive the pipeline with a deterministic in-memory system and stub checkers. These
are test doubles for the composition logic, not shipped data: no test fabricates a dataset, a
benchmark, or a result. The experiments themselves are not unit-tested, because they require the
real datasets.

---

## Scope of the guarantees

Stated plainly, because the framework claims less than it might appear to.

An execution certified by all five guards satisfies the three exactly decided clauses
deterministically, and the two learned clauses with a probability of wrongful certification of at
most `2α` at confidence at least `1 − 2δ`, by a union bound. No stronger statement holds.

The framework does **not** prove that a model is correct or uncompromised. The attestation layer
does not detect a manipulation that preserves the complete fingerprint, a model that was already
compromised when the baseline was recorded, or an attack that subverts the trusted computing base.
The first two belong to offline attestation; the third is excluded by the trust model. Attack
family A4 exists precisely to probe that boundary, and it is one adaptive construction rather than
a worst case.

The trusted computing base is the guards, the append-only log, and the human reviewer. The governed
model is outside it and is not trusted for integrity.

---

## Citation

```bibtex
@article{hassan2026governance,
  title   = {An {AI} Governance Pipeline Framework with Trustworthy {AI} Practice
             for High-Risk {AI} Systems},
  author  = {Hassan, Muhammad and Islam, Shareeful},
  journal = {},
  year    = {2026}
}
```

---

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

The datasets and benchmarks are **not** covered by this licence and remain subject to their own
terms. MIMIC-IV requires PhysioNet credentialed access, the i2b2/UTHealth corpus requires its own
data-use agreement, and the Fraunhofer-Beads pilot data cannot be redistributed.
