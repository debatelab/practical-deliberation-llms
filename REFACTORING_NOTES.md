**Phase 0 – Clarify Target Behavior**

1. Decide the scoring semantics:
   - We follow the plan literally:
     - Use `/v1/chat/completions`.
     - Use `structured_outputs` with `structural_tag`.
     - Force output format `<think>{r}</think>{"label": "<LABEL>"}`.
     - Restrict `label` to a finite set `labels = ["a", "b", ...]`.
     - Read `top_logprobs` at the *first* label token and map them to `P(label | context, r)` via `logprobs_to_label_probs`.
   - IMPORTANTLY: We score *conditioned on the original trace*:
     - Reasoning text `r` comes from the `trace["think"]` field produced in `experiments/re_sampling_stability/reasoning.py`.
     - Scoring must use that `r`, not regenerate a new one.

---

**Phase 1 – API Changes in `InferenceClient`**

3. Redesign `InferenceClient.score_label_given_trace` API:
   - Current signature:  
     `score_label_given_trace(self, prompt: str, labels: List[str]) -> Dict[str, float]`
   - Target signature (one reasonable option):  
     `score_label_given_trace(self, context_messages: List[dict], reasoning: str, labels: List[str]) -> Dict[str, float]`
     - `context_messages`: chat messages describing decision situation, options, and scoring instructions, **excluding** `r`.
     - `reasoning`: the fixed reasoning trace `r` (string) to enforce via grammar.
     - `labels`: list of allowed labels (`["a", "b", ...]`).

4. Update all call sites:
   - In `experiments/re_sampling_stability/judgment.py`, change how you call the scoring method:
     - Instead of building a *single flat prompt string* via `SCORE_PROMPT_TEMPLATE`, RE-build or RETRIEVE `context_messages` from reasoning phase (likely a `system` + `user` pair), using exactly the same propmt templates that were used there.
     - Pass `trace["think"]` (or fallback) as the `reasoning` argument.
     - Pass the same `label_letters` list as `labels`.

---

**Phase 2 – Build `structured_outputs` Grammar JSON**

6. Implement a helper to construct the `structural_tag` JSON:
   - Place it either:
     - In `InferenceClient` (private method), or
     - In a small dedicated module (e.g. `practical_deliberation_llms/grammar.py`) if you want to reuse it elsewhere.
   - Based on XGrammar docs and the plan, shape should be roughly:

     ```python
     def make_structured_label_grammar(reasoning: str, labels: List[str]) -> dict:
         return {
             "structured_outputs": {
                 "structural_tag": {
                     "type": "structural_tag",
                     "format": {
                         "type": "sequence",
                         "elements": [
                             {
                                 "type": "tag",
                                 "begin": "<think>",
                                 "content": {
                                     "type": "const_string",
                                     "value": reasoning,
                                 },
                                 "end": "</think>",
                             },
                             {
                                 "type": "json_schema",
                                 "json_schema": {
                                     "type": "object",
                                     "properties": {
                                         "label": {
                                             "type": "string",
                                             "enum": labels,
                                         },
                                     },
                                     "required": ["label"],
                                     "additionalProperties": False,
                                 },
                             },
                         ],
                     },
                 },
             },
         }
     ```


7. Make sure the grammar conforms to XGrammar docs:
   - `structural_tag` root.
   - `sequence` of `tag` then `json_schema`.
   - `enum` on `label` matches allowed labels exactly (case, quotes).

---

**Phase 3 – Logprobs Extraction**

8. Prepare the chat request parameters in `InferenceClient.score_label_given_trace`:
    - Use the OpenAI Python SDK’s chat endpoint:

      ```python
      extra_body = make_structured_label_grammar(reasoning, labels)

      response = self.client.chat.completions.create(
          model=self.model,
          messages=context_messages,
          temperature=0.0,
          top_p=1.0,
          logprobs=True,
          max_tokens=...,  # see next step
          extra_body=extra_body,
      )
      ```

    - Decide `max_tokens`:
      - Must be sufficient for:
        - `<think>`
        - Tokenization of `reasoning`
        - `</think>`
        - JSON snippet (`{"label": "X"}`)
      - A safe starting point: `max_tokens = len(reasoning) // 2 + 32` or just a fixed upper bound (e.g. 512) if you don’t mind over‑allocating.

9. Determine label token step (`step_label_start`):
    - From the plan, you need the index where the label’s first token is produced.
    - One practical method:

      - Get generated text (assistant output) as a string.
      - Find the offset of the substring that starts right before the label: e.g. after `{"label": "`.
      - Use the token‑by‑token incremental reconstruction from `logprobs` to locate the token index where this substring begins.
      - Pseudosteps:
        - Initialize `accum = ""`.
        - Iterate over generation steps, appending the chosen token at each step (from `choices[0].logprobs.tokens` or from the text slicing if available).
        - When `accum` ends with `{"label": "`, record that step index as `step_label_start`.
      - This aligns with the plan’s suggestion to “detokenize incrementally until we pass `{"label": "`”.

    - If vLLM’s chat `logprobs` format exposes tokens directly per step (similar to completions), you can rely on that structure / docs.

10. Extract `top_logprobs` at that step:
    - Once you have `step_label_start`, take:

      ```python
      top_logprobs_step = choice.logprobs.top_logprobs[step_label_start]
      ```

      (exact attribute path may differ; adapt to vLLM’s actual structure.)

    - This should be a list of dicts like `{"token": str, "logprob": float, ...}`.

11. Convert to label probabilities:
    - Call the existing helper from `src/practical_deliberation_llms/util.py`:

      ```python
      label_probs = logprobs_to_label_probs(top_logprobs_step, labels)
      ```

    - Keep the logging in `InferenceClient.score_label_given_trace` so it’s easier to debug.

---

**Phase 4 – Integrate with Experiment Pipeline**

12. Update `score_choice_labels_for_trace` to use structured scoring:
    - Change it to prepare `context_messages` and call the new `InferenceClient` method:

      ```python
      context_messages = build_scoring_messages(decision_situation, labeled_actions, think)
      label_probs = inference_client.score_label_given_trace(
          context_messages=context_messages,
          reasoning=think or "",
          labels=label_letters,
      )
      ```

    - Retain its downstream behavior:
      - Normalize probabilities to sum to 1.
      - Fall back to uniform distribution if total mass is 0.
      - Attach metadata (`problem_uid`, `trace_id`, `candidate_model`, `assistant_model`, etc.).

13. Ensure trace data includes `think`:
    - It already does in `experiments/re_sampling_stability/reasoning.py` (using `InferenceClient.generate_trace`).
    - Still, handle missing `think`:
      - If `think` is `None`, either:
        - Skip structured scoring and fall back to legacy scoring.
    - However, take care not to construct constraint that accidentally enforces, in step 6 above, "<think><think>...</think></think>"
    - Insert generous logging for later debugging

16. Keep configuration defaults coherent:
    - `experiments/re_sampling_stability/configs/*.yaml` currently point `score_choice_labels_fn` to `experiments.re_sampling_stability.judgment.score_choice_labels_for_trace`.
    - Make sure the function signature in `judgment.py` is unchanged so YAML doesn’t need edits.
    - If you add a feature flag (e.g. `use_structured_scoring`), extend `ExperimentConfig` / `CLIConfig` accordingly and plumb it through `run_experiment.py`.



# APPENDIX


QUESTIONS:
> 
> 1) Exact prompts for the scoring step**
> Questions:
> 
> 1) Do you want me to:>  - Completely retire `SCORE_PROMPT_TEMPLATE`, and
- Use exactly the same `SYSTEM_PROMPT_TEMPLATE` and `USER_PROMPT_TEMPLATE` as the context messages for scoring (no scoring-specific text), relying entirely on the > structured grammar to enforce the `<think>…</think>{"label": ...}` behavior?

YES 

> 2) Or do you want a *separate* scoring-specific system/user prompt pair (e.g. with slightly different instructions) that we define once and reuse for scoring only?

NO

> **2) Location and design of the grammar helper**
> 
> The plan suggests either:
> 
> - A private helper on `InferenceClient`, or
> - A small dedicated module, e.g. `practical_deliberation_llms/grammar.py`.
> 
> Questions:
> 
> 3) Do you have a preference between:>  - `InferenceClient._make_structured_label_grammar(reasoning, labels)`, or>  - A shared `practical_deliberation_llms.grammar.make_structured_label_grammar(...)` that `InferenceClient` just calls?

I PREFER a standalone `grammar.py` MODULE.

4) Do you want a way to toggle between the strict `const_string` reasoning constraint and a relaxed `any_text` version (as described near lines 460–470 of > `PLAN_COLAB_NOTEBOOK.md`), e.g.:
> 
> - A boolean parameter on `score_label_given_trace` like `enforce_exact_reasoning: bool = True`, or
> - A config flag on `ExperimentConfig`?
> 

No configs for that, don't implement `any_text` at all.
Instead: HARD-CODE `const_string`, keeping the code structured.



> 5) Is it acceptable to drop the old `prompt`-based signature entirely (and update the `DummyInferenceClient` in `tests/test_run_experiment_async.py` accordingly)

YES. drop the old `prompt`-based signature entirely

> , or do > you want a small backward-compat shim like:

NO, no shims

> 
> **4) Feature flag vs hard switch to structured scoring**
> Questions:
> 
> 6) For the experiments under `experiments/re_sampling_stability`, do you want:
> 
> - A hard switch: always use structured scoring (chat + `structured_outputs`); the old `/v1/completions`-based scoring is effectively retired, or

YES

> - A config-controlled switch on `ExperimentConfig` (and optionally CLI/ YAML), e.g.:

NO

> 
> **5) Behavior when the trace is missing `think`**
> 
> Currently:
> 
> - `reasoning.py` logs warnings when `think` is missing but still produces a trace dict.
> - `judgment.py` blindly includes `think` (possibly `None`) in `SCORE_PROMPT_TEMPLATE`.
> 
> Questions:
> 
> 7) What should the new behavior be when `trace["think"]` is missing or empty?
> 
> Options:
> 
> 1. Skip scoring for that trace entirely (and log a warning).

YES

> 2. Fall back to the legacy flat-prompt scoring (still via `score_label_given_trace`, but bypassing structured grammar).
  NO
> 3. Use structured grammar with `reasoning=""` (effectively “no reasoning”) and still score labels.
  NO

> **6) Labels: assumptions and constraints**
> 
> Right now:
> 
> - Experiments derive labels as `["a", "b", ...]` in both `reasoning.py` and `judgment.py`.
> - `logprobs_to_label_probs` maps tokens to labels by substring matching (case insensitive).
> 
> Questions:
> 
8) Can we assume for this refactor that labels for scoring are *always* simple one-character strings (`"a"`, `"b"`, …) as generated today, or do you want the new code to > support arbitrary label strings (e.g. `"A"`, `"B"`, `"C"`, or `"yes"`, `"no"`) passed in via `labels`?
> 
If you want generality, I’ll slightly tighten `logprobs_to_label_probs` (e.g. prefer exact-equality matches on `token.strip('"').lower()` before falling back to substring> ) to be more robust to structured_outputs tokenization.

YES, that is a good idea. Please implement this.

> **7) Model choice for scoring vs reasoning**
> 
> In `run_experiment_async`, we currently construct:
> 
> ```python
> inference_client = InferenceClient(client=client, model=config.candidate_model)
> ```
> 
> Both trace generation and scoring use that same `candidate_model`, while `assistant_model` is just stored as metadata.
> 
> Questions:
> 
9) For this refactor, do you want to keep that behavior (scoring and reasoning use the same `candidate_model`)?

YES, keep the behavior
