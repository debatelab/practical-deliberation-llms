**1) Settle Experiment Package Layout**

- [+] Create a real Python package for the experiment code:
  - e.g. `experiments/re_sampling_stability` (underscore),
- [+] Add empty files (stubs) in `experiments/re_sampling_stability`:
  - `transform.py`
  - `reasoning.py`
  - `judgment.py` 
- [+]Make sure the dotted paths used in the CLI defaults:
  - `experiments.re_sampling_stability.transform.transform_problems`
  - `experiments.re_sampling_stability.reasoning.generate_reasoning_traces_for_problem`
  - `experiments.re_sampling_stability.judgment.score_choice_labels_for_trace`
  actually resolve.
- Note: For now, experiment packages are intended primarily for CLI use;
  they are importable as namespace packages when the repo is on the
  Python path, but there is no additional support for notebook imports.


**2) Implement The Three Async Pluggable Functions**

- [+] `transform.transform_problems(config, problem) -> list[problem]`:
  - Default behavior: include base problem;
  - implement a basic `reverse_options` variant (respect `config.max_transformations_per_problem`).
  - Attach transformation metadata as in the notebook (`base_problem_uid`, `transformation_type`, etc.).
  - Watch out: preserve or deterministically create `problem_uid`s for variants.
- [+] `reasoning.generate_reasoning_traces_for_problem(config, inference_client, problem) -> list[dict]`:
  - Mirror notebook step 7; use Jinja prompts; call `InferenceClient.generate_trace`.
  - Return dicts with a stable set of keys (`problem_uid`, `trace_id`, `decision_situation`, `actions`, `labels`, `think`, etc.).
  - Watch out: consistent seeding strategy and `trace_id` handling; avoid leaking pandas/HF details in here.
- [+] `judgment.score_choice_labels_for_trace(config, inference_client, trace) -> list[dict]`:
  - Mirror notebook step 8; reconstruct prompt with reasoning; call `InferenceClient.score_label_given_trace`.
  - Return one dict per label with (`problem_uid`, `trace_id`, `label`, `prob`, plus transformation metadata).
  - Watch out: ensure probabilities are normalized and handle degenerate cases (all-zero probabilities → uniform).

**3) Define Minimal Trace/Score “Schemas”**

- [+] Even without dataclasses, write down in docstrings/comments the expected keys for:
  - trace dicts (output of reasoning function),
  - score dicts (output of scoring function).
- [+] Make `run_experiment.py` comments explicit about what `compute_metrics` and `save_results` will expect later.
- [+] Keep these signatures loose until metrics/plots stabilize. Consider whether to add dataclasses later.

**4) Implement Fixed Analysis / Plot / Save Helpers**

- [+] `experiments/re_sampling_stability/metrics.py`:
  - `compute_metrics(trace_records: list[dict], score_records: list[dict]) -> (d_within_df, baseline_vs_trans_df)` or accept DataFrames.
  - Use `within_context_disagreement` and `kl_divergence` from `util` (may need to implement those there).
- [+] `src/practical_deliberation_llms/io.py`:
  - `save_results(config, output_dir, trace_df, scores_df, d_within_df, baseline_vs_trans_df)`.
- [+] `src/practical_deliberation_llms/plotting.py`:
  - `plot_results(output_dir, d_within_df, baseline_vs_trans_df)`.
- [+] Then replace the TODO block in `run_experiment_async` with real calls.
- [+] Watch out: import paths (avoid circular imports); ensure these modules do not depend on experiment-internal async logic.

**5) Solidify Core Library Support (`util`, `inference`)**

- [+] In `practical_deliberation_llms.util`:
  - Implement: `logprobs_to_label_probs`, `parse_think_and_label`, `extract_label_from_json`, `kl_divergence`, `within_context_disagreement`.
  - Keep `get_labelprobs_from_message` working for any existing notebooks.
- [ ] In `practical_deliberation_llms.inference` (manual integration check):
  - Confirm `InferenceClient.generate_trace` and `score_label_given_trace` work end-to-end with your vLLM/OpenAI server. This requires a running server and is intentionally left as a manual step.
  - Longer term: move `score_label_given_trace` to `chat.completions` / structured outputs; not critical for the first experiment script, but important for consistency.
- [ ] Things to watch:
  - Logprob shapes from vLLM can differ from OpenAI’s; test this path early.
  - Parsing `<think>...</think>` and JSON must be robust to model “creativity”.

**6) Decide On Concurrency Model (Later)**

- [ ] Once correctness is established and the baseline async flow is stable, consider:
  - Batching reasoning and scoring with `asyncio.gather` for better throughput.
  - Maybe a simple concurrency limit (semaphore) to avoid hammering vLLM.
- [ ] Watch out: reproducibility with concurrency + seeding, and logging that remains readable.

**7) Testing And Example Configs**

- [+] Add at least:
  - A minimal YAML example (no transform, small `n_problems`).
  - A YAML example with a transformation function configured.
- [+] Write small tests or scripts that:
  - Call `build_config` with and without YAML (including empty string and `null` for `transform_problems_fn`).
  - Run `run_experiment_async` with dummy pluggable functions (no real inference) to validate control flow and that the skip-transform logic behaves as intended.


**8) Replacing `config.py` With chz**

Here is a concrete way to move from the current ad‑hoc `dataclasses + argparse + YAML` setup to `chz` while preserving (or improving) behavior.

---

**1. What You Have Today**

The current `ExperimentConfig` system in `experiments/re_sampling_stability/config.py` does three main things:

- Models config:
  - `DatasetSpec` and `ExperimentConfig` as `@dataclass`es.
- Loads config:
  - Reads optional YAML (`--config-yaml`) into `cfg_from_yaml`.
  - Enforces that `datasets` is a non‑empty list of mappings with `name`, `n_problems`, etc.
- Merges CLI and YAML:
  - CLI > YAML > argparse defaults via `resolve_field`.
  - Special handling for:
    - `transform_problems_fn`: empty string → `None`.
    - `make_plots`: `--no-plots` boolean.
    - `output_dir` (with TODO about timestamped subdir).
  - Environment interplay: `openai_base_url` and `api_token` can be provided, otherwise you rely on env vars.

`run_experiment.py` (not shown, but implied) consumes a ready‑made `ExperimentConfig`.

---

**2. Relevant chz Features**

From the chz docs:

- 02_object_model:
  - `@chz.chz` gives you an immutable, keyword‑only, dataclass‑like config object with validation hooks (`chz.field`, `@chz.init_property`).
- 04_command_line:
  - `chz.entrypoint` or `chz.nested_entrypoint` turns a class/function into a CLI, with automatic `--help`, type‑aware casting, nested objects, etc.
- 05_blueprint:
  - `chz.Blueprint(ConfigClass)` lets you apply partial config data (dicts, nested dicts) in stages; later `.apply()` calls override earlier ones → perfect for “YAML base, CLI override”.
- 06_serialisation:
  - `beta_to_blueprint_values` / `asdict` give you dicts that match what `Blueprint.apply()` expects; useful for saving/loading configs.

These map pretty directly onto what `config.py` is doing by hand.

---

**3. Target Design With chz**

The high‑level idea:

- Represent `DatasetSpec` and `ExperimentConfig` as `@chz.chz` classes instead of `dataclasses.dataclass`.
- Use `chz.entrypoint` (or `chz.nested_entrypoint`) to parse CLI into either:
  - a full `ExperimentConfig`, or
  - a small “launcher” config that combines a YAML file and CLI overrides into a final `ExperimentConfig`.
- Use `Blueprint` to implement the “YAML + CLI with CLI precedence” behavior.

Step‑by‑step:

1) Model config as chz classes

- Replace:

  ```python
  @dataclass
  class DatasetSpec:
      name: str
      n_problems: int
      adapter_kwargs: Dict[str, Any]
  ```

  with something like:

  ```python
  @chz.chz
  class DatasetSpec:
      name: str
      n_problems: int
      adapter_kwargs: dict[str, object] = chz.field(default_factory=dict)
  ```

- Replace `ExperimentConfig` similarly:

  ```python
  @chz.chz
  class ExperimentConfig:
      candidate_model: str
      assistant_model: str
      openai_base_url: str | None = None
      api_token: str | None = None

      seed: int
      datasets: list[DatasetSpec]

      n_traces_per_problem: int
      temperature: float
      top_p: float

      max_transformations_per_problem: int

      output_dir: str
      make_plots: bool = True

      transform_problems_fn: str | None = None
      generate_reasoning_trace_fn: str
      score_choice_labels_fn: str
  ```

- Use `chz.field` and validation (from 03_validation / 22_field_api) to enforce invariants you currently check in `build_config`:
  - Non‑empty `datasets`
  - `n_problems > 0`
  - `n_traces_per_problem > 0`, etc.

  For example:

  ```python
  from chz.validators import gt

  @chz.chz
  class DatasetSpec:
      name: str
      n_problems: int = chz.field(validator=gt(0))
      adapter_kwargs: dict[str, object] = chz.field(default_factory=dict)
  ```

2) Move env‑var defaults into config object

- Instead of relying on external env handling, you can embed it in an `init_property` (02_object_model / 21_post_init):

  ```python
  @chz.chz
  class ExperimentConfig:
      openai_base_url: str | None = None
      api_token: str | None = None

      @chz.init_property
      def effective_openai_base_url(self) -> str | None:
          return self.openai_base_url or os.getenv("OPENAI_BASE_URL")

      @chz.init_property
      def effective_api_token(self) -> str | None:
          return self.api_token or os.getenv("OPENAI_API_KEY")
  ```

- Downstream code can then use `config.effective_openai_base_url` instead of re‑implementing fallback logic.

3) Preserve “YAML base, CLI override” with Blueprint

Design a “launcher” config whose job is to specify the YAML file and optional overrides:

```python
@chz.chz
class ExperimentOverrides:
    candidate_model: str | None = None
    assistant_model: str | None = None
    openai_base_url: str | None = None
    api_token: str | None = None
    seed: int | None = None
    n_traces_per_problem: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_transformations_per_problem: int | None = None
    output_dir: str | None = None
    make_plots: bool | None = None
    transform_problems_fn: str | None = None
    generate_reasoning_trace_fn: str | None = None
    score_choice_labels_fn: str | None = None
```

And then a CLI “envelope”:

```python
@chz.chz
class LauncherConfig:
    config_yaml: str  # path to YAML (required)
    overrides: ExperimentOverrides = chz.field(default_factory=ExperimentOverrides)
```

CLI entrypoint (04_command_line, 01_quickstart):

```python
def main(cfg: LauncherConfig) -> None:
    # 1. Load YAML into a plain dict
    with open(cfg.config_yaml, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f) or {}

    # 2. Apply YAML as base config
    bp = chz.Blueprint(ExperimentConfig)
    bp.apply(yaml_data)

    # 3. Apply CLI overrides on top (later apply wins)
    override_dict = {k: v for k, v in chz.asdict(cfg.overrides).items() if v is not None}
    bp.apply(override_dict)

    experiment_cfg = bp.make()

    # 4. Call existing experiment runner
    run_experiment(experiment_cfg)
```

Then:

```python
if __name__ == "__main__":
    chz.nested_entrypoint(main)
```

This gives you:

- YAML as base config.
- CLI overrides expressed as `overrides.seed=123`, `overrides.output_dir="/tmp/foo"`, etc.
- CLI > YAML precedence implemented naturally via `Blueprint.apply`.

4) Handling `datasets` in YAML and CLI

- YAML already holds `datasets` as a list of mappings; chz will happily cast that into `list[DatasetSpec]` when you call `Blueprint.apply(yaml_data)` as long as the shapes match.
- If you need CLI overrides for datasets, chz’s nested argument syntax (04_command_line) lets you do things like:

  - `datasets[0].n_problems=20`
  - or use presets / partial blueprints if you want more ergonomic options.

- To preserve the current “YAML must define non‑empty datasets” behavior, either:
  - Keep that validation in `build_config`‑equivalent logic before constructing the blueprint, or
  - Move it to class‑level `@chz.validate` on `ExperimentConfig`.
