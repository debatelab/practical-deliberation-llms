Goal: Create a Colab Notebook running on free GPU that can be used to check the reason-responsiveness of small LLMs in customizable experiments

Sub-Goal: Create utils and models that allow one to setup clean and simple notebook.

Macro Workflow / Structure of Notebook:

1. Choose parameters / configure experiment

- model to test
- dataset with decision problem, sample size
  - https://huggingface.co/datasets/kellycyy/daily_dilemmas
  - https://isir-wuya.github.io/Multi-step-Moral-Dilemmas/
  - https://www.kaggle.com/datasets/jianloongliew/reddit/data & https://github.com/JianLoong/reddit-store & https://huggingface.co/datasets/derek-thomas/dataset-creator-reddit-amitheasshole
  - https://github.com/ddindidu/RoleConflictBench
  - [kellycyy/AIRiskDilemmas](https://huggingface.co/datasets/kellycyy/AIRiskDilemmas)
  - 

- OPTIONAL: transformation to apply to decision scenario to test judgment stability (no reason-responsiveness without judgment stability)
  - requires: model for context transformations
- edit decision-making / judgment prompt

1. Load model to test

- load model, serve with vllm

2. Load dataset with decision scenarios (needs to be multiple-choice Q/A)

- load dataset
- sample subset of size N
- postprocess (format: context, question, options)

3. Apply transformations

- Apply transformations to decision scenarios to generate, for each scenario, up to k variations of `context`
- store variations in dataset

4. Generate judgments (Q/A with reasoning)

- Let model generate judgments for baseline (and all variations)
  - sampling multiple chain of thought traces,m possibly with minimally changed sampling parameters
  - elicit confidence in each choice (logprobs!)
  - decision logprobs should not be inavriant!

5. Analyse and visualize results

- generate table and possibly graph to illustrate how stable judgments are (depending on type of variation and compared to baseline models)
