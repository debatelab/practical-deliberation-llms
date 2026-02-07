# Probing Deliberation Skill via Choice Disposition Robustness to IID-Sampled Reasoning Traces

This module runs judgment-stability experiments for decision problems with
multiple options. It probes how stable a model's *choice disposition* is when
we re-sample its internal reasoning traces and (optionally, not-yet-implemented) re-sample the
presentation of the same underlying dilemma.

The code lives in `experiments/re_sampling_stability` and is wired up via
`run_experiment.py`.

## Theoretical Background

We model a decision problem as a pair $P = \langle S, C \rangle$, where

- $S$ is a natural-language description of a situation, and
- $C = \{c_1, \dots, c_k\}$ is a finite set of options.

We fix a stochastic **deliberator** (e.g. an LLM with sampling enabled) with
parameters $\theta$. For a given problem $(S, C)$, repeated calls to this
deliberator with the same input and non-zero temperature yield different
reasoning traces $r$ (we treat these as i.i.d. samples of internal reasoning).

Given a particular trace $r$, we view the deliberator as inducing a stochastic
**policy** over options, denoted $\pi_\theta(c \mid S, C, r)$, which gives the
probability that the deliberator ultimately selects option $c \in C$ after
"thinking" along $r$.

For convenience, we write

$$
Q_j(c_i) = \pi_\theta(c_i \mid S, C, r_j)
$$

for the choice disposition induced by the $j$-th sampled trace $r_j$.

Our working **deliberative competence assumption** is:

> If an agent can competently deliberate about problem $P$, then i.i.d. samples
> $r_1, \dots, r_n$ of its reasoning for $P$ should induce similar policies
> $\pi_\theta(\cdot \mid S, C, r_j)$ over options (equivalently, similar
> choice dispositions $Q_j(\cdot)$ ).

Concretely, for each problem we:

1. Sample $n$ independent traces $r_1, \dots, r_n$ by repeatedly calling the
   deliberator with the same $(S, C)$ and sampling enabled.
2. For each trace $r_j$, estimate the induced policy
   $\pi_\theta(c_i \mid S, C, r_j)$ over options $c_i \in C$ via a second
   "scoring" call, and record it as $Q_j(c_i)$.
3. Measure the dispersion of $\{Q_1, \dots, Q_n\}$ using the
   Jensen–Shannon information radius (`compute_jsd_information_radius`).

Low information radius corresponds to stable choice dispositions under
re-sampling of reasoning; high radius indicates that the model's choice is
highly sensitive to which trace happened to be sampled.

We also (_plan to_) study robustness under **presentation re-sampling**: applying
transformations that preserve the underlying decision problem but change its
surface form (e.g. permuting the order of actions). For each base problem
and transformed variant, we compare the mean choice dispositions
$\bar{Q}_\text{base}$ and $\bar{Q}_\text{trans}$ using KL divergence
(`kl_divergence`).


## Related Work

* Wang, Xuezhi, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. "Self-consistency improves chain of thought reasoning in language models." arXiv preprint arXiv:2203.11171 (2022).


## Experimental Design

The main entry point is `run_experiment.py`, which orchestrates a complete
experiment run given an `ExperimentConfig` (see `config.py`). For each run we:

1. **Sample decision problems** using
   `practical_deliberation_llms.datasets.sample_problems` according to the
   `datasets` field in `ExperimentConfig`.
2. **(Optional) Transform problems** by applying an async transform function
   to each base problem (e.g. `transform.permutate_actions`), always
   including the original problem.
3. **Generate reasoning traces** via
   `reasoning.generate_reasoning_traces_for_problem`, which calls
   `InferenceClient.generate_trace` to obtain `<think>...</think>` content and
   a JSON label for each of `n_traces_per_problem` samples.
4. **Score choice labels** via
   `judgment.score_choice_labels_for_trace`, which conditions on the original
   dilemma and the sampled reasoning $r_j$ to estimate the policy
   $\pi_\theta(c_i \mid S, C, r_j)$ over options (recorded as
   per-trace distributions $Q_j(c_i)$).
5. **Compute metrics and plots** using `metrics.compute_metrics`,
   `save_results`, and `plot_results` (in the top-level
   `practical_deliberation_llms` package).

## Running the Experiment

Configuration is driven by a YAML file plus optional CLI overrides. Typical
usage (from the repository root):

```bash
uv run python -m experiments.re_sampling_stability.run_experiment \
  config_yaml=experiments/re_sampling_stability/configs/minimal.yaml
```

To override the model or plug in custom functions:

```bash
uv run python -m experiments.re_sampling_stability.run_experiment \
  config_yaml=experiments/re_sampling_stability/configs/with_transform.yaml \
  candidate_model=Qwen/Qwen2.5-7B-Instruct \
  generate_reasoning_trace_fn=\
    experiments.re_sampling_stability.reasoning.generate_reasoning_traces_for_problem \
  score_choice_labels_fn=\
    experiments.re_sampling_stability.judgment.score_choice_labels_for_trace
```

Results are written to `ExperimentConfig.output_dir` in JSONL or Parquet
format (see `ExperimentConfig.results_file_format`).

## Outputs

A typical run produces the following files under the configured `output_dir`:

- `traces.*` – one row per reasoning trace (metadata and text).
- `scores.*` – one row per (trace, label) with
  `prob ≈ π_θ(c_i | S, C, r_j)` (the estimated policy value, i.e. $Q_j(c_i)$).
- `metrics_within.*` – per-problem within-context JSD information radius and
  trace counts.
- `baseline_vs_trans.*` – per base problem, KL divergence between
  $\bar{Q}_\text{base}$ and each transformed variant.
- `within_disagreement_hist.png` – histogram of within-context dispersion.

These artifacts are the primary inputs for downstream analysis in notebooks
and papers.
