# Activation Stance Classifier

A research framework for **analyzing and steering political stance** in Large Language Model (LLM) activations without fine-tuning. This project implements an end-to-end pipeline from activation extraction to bias intervention and evaluation.

## Research Overview

This project investigates how political bias manifests in LLM internal representations and develops methods to **steer model behavior** through targeted activation interventions. The approach:

1. **Extracts** neuron activations from politically-labeled text
2. **Identifies** politically-relevant neurons using SVM feature selection
3. **Optimizes** intervention multipliers to shift model political stance
4. **Evaluates** the intervention's effect using Likert-scale questionnaires
5. **Validates** that interventions don't harm general capabilities (PoETa benchmark) *(deprecated)*

## Pipeline Architecture

The project uses **W&B Artifacts** for full reproducibility and lineage tracking:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          W&B ARTIFACT PIPELINE                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. EXTRACTION                                                               │
│     extract_activations.py                                                   │
│     └──► activations-{dataset}-{model}-{layers}:latest                       │
│                            │                                                 │
│                            ▼                                                 │
│  2. FEATURE SELECTION                                                        │
│     train_eval_svc.py                                                        │
│     └──► svm-feature-ranking-{activations_artifact}:latest                   │
│                            │                                                 │
│                            ▼                                                 │
│  3. OPTIMIZATION                                                             │
│     optimize_intervention.py                                                 │
│     └──► {study_name}:latest  (intervention multipliers)                     │
│                            │                                                 │
│              ┌─────────────┴─────────────┐                                   │
│              ▼                           ▼                                    │
│  4. LIKERT PI EVALUATION                                                     │
│     likert_scale_test.py              likert_scale_test.py                   │
│     (baseline — no intervention)      (with intervention multipliers)        │
│     └──► likert-baseline-results      └──► likert-intervened-results         │
│              │                           │                                   │
│              └───────────┬───────────────┘                                   │
│                          │  (comparison visualizations are generated         │
│                          │   automatically via plot_pi_shift.py)             │
│                          ▼                                                   │
│  5. POETA EVALUATION  ⚠️ deprecated                                          │
│     poeta_evaluator.py                                                       │
│     └──► PoETa V2 benchmark (baseline vs intervened)                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
├── src/                              # Core Python modules
│   ├── extract_activations.py       # Step 1: Extract model activations
│   ├── train_eval_svc.py            # Step 2: Train SVM, select features
│   ├── optimize_intervention.py     # Step 3: Optuna optimization
│   ├── likert_scale_test.py         # Step 4: Polarization Index evaluation
│   ├── poeta_evaluator.py           # Step 5: PoETa benchmark (deprecated)
│   ├── compile_target_neurons.py    # Helper: Select neurons from ranking
│   ├── model_factory.py             # Model loading (Llama/Gemma)
│   ├── llama_3dot1_wrapper.py       # Llama 3.1 wrapper (TransformerLens)
│   ├── gemma_3_wrapper.py           # Gemma 3 wrapper (TransformerLens)
│   └── activation_df.py             # Activation data utilities
├── visualizations/
│   ├── plot_pi_shift.py             # PI shift comparison plots
│   ├── create_triple_comparison_wandb.py  # Base/Max/Min composite from W&B artifacts
│   └── multipliers.py              # Multiplier distribution boxplot
├── scripts/
│   ├── create_activation_datasets.sh  # Batch extraction runner
│   ├── train_eval_SVCs.sh             # Batch SVM training runner
│   ├── run_optimization.sh            # Optimization launcher
│   ├── test_poeta_eval.sh             # Quick PoETa smoke test
│   ├── log_dataset_to_wandb.py        # Upload dataset as W&B artifact
│   └── clear_optuna_run.py            # Delete Optuna study from SQLite
├── config/                           # Hydra configuration files
│   ├── config.yaml                  # Base defaults (extraction + SVM)
│   ├── llama-3.1-8b.yaml           # Unified Llama 3.1 8B config (all steps)
│   ├── gemma-3-4b.yaml             # Unified Gemma 3 4B config (all steps)
│   ├── gemma-3-27b.yaml            # Unified Gemma 3 27B config (all steps)
│   ├── optimization_llama.yaml      # Llama optimization config
│   ├── optimization_gemma.yaml      # Gemma optimization config
│   ├── likert_eval-llama.yaml       # Llama Likert evaluation config
│   ├── likert_eval-gemma.yaml       # Gemma Likert evaluation config
│   ├── poeta_eval.yaml              # PoETa benchmark config (deprecated)
│   └── poeta_eval_compare.yaml      # PoETa comparison config (deprecated)
├── likert/
│   └── questions_anderson.csv       # Likert questionnaire (paired P+/P-)
├── data/                             # Activation datasets (parquet)
├── artifacts/                        # Downloaded W&B artifacts cache
├── runs/                             # Experiment outputs (Hydra run dirs)
└── wandb/                            # W&B tracking logs
```

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases
wandb login
```

### Full Pipeline Execution

Each model has a **unified config** (`llama-3.1-8b.yaml`, `gemma-3-4b.yaml`, `gemma-3-27b.yaml`, `qwen-3-8b.yaml`, `qwen-3-0.6b.yaml`, `phi-3.yaml`) that can be used across all pipeline steps. Alternatively, use the step-specific configs.

> TODO: Update step 4; baseline is automatically executed within the Likert Scale Test to generate comparison figs
```bash
# 1. Extract activations from your dataset
python src/extract_activations.py --config-name llama-3.1-8b \
  data.input_csv="/path/to/political_texts.csv" \
  extraction.layers="all"

# 2. Train SVM and identify politically-relevant neurons
python src/train_eval_svc.py --config-name llama-3.1-8b \
  data.activations_artifact_name="activations-political_texts-llama-all:latest"

# 3. Optimize intervention multipliers
python src/optimize_intervention.py --config-name llama-3.1-8b \
  optimization.feature_artifact_name="svm-feature-ranking-activations-political_texts-llama-all:latest" \
  optimization.direction="minimize"  # Shift LEFT

# 4a. Run baseline Likert evaluation (no intervention)
python src/likert_scale_test.py --config-name likert_eval-llama \
  likert.multiplier_artifact_name=null

# 4b. Run intervened Likert evaluation
python src/likert_scale_test.py --config-name likert_eval-llama \
  likert.multiplier_artifact_name="intervention-multipliers:latest"

# 5. (Deprecated) PoETa capability benchmark
python src/poeta_evaluator.py --config-name poeta_eval \
  compare_baseline=true
```

## Configuration

All scripts use [Hydra](https://hydra.cc/) for configuration management. Override any parameter via CLI:

```bash
# Use a unified model config for any step
python src/extract_activations.py --config-name gemma-3-4b

# Override specific parameters
python src/optimize_intervention.py --config-name llama-3.1-8b \
  optimization.direction="maximize" \
  optimization.n_trials=500

# Change dataset
python src/extract_activations.py --config-name llama-3.1-8b \
  data.input_csv="/path/to/different_data.csv"
```

### Config Files

| File | Purpose |
|------|---------|
| `config.yaml` | Base defaults for extraction and SVM training |
| `llama-3.1-8b.yaml` | **Unified** Llama 3.1 8B config (all pipeline steps) |
| `gemma-3-4b.yaml` | **Unified** Gemma 3 4B config (all pipeline steps) |
| `gemma-3-27b.yaml` | **Unified** Gemma 3 27B config (all pipeline steps) |


## Key Concepts

### Polarization Index (PI)

The **Polarization Index** measures model political stance on a scale of **[-4, +4]**:
- **Positive PI**: Model agrees more with right-leaning statements
- **Negative PI**: Model agrees more with left-leaning statements
- **Zero PI**: Neutral/balanced responses

Computed from paired questions (P+ right-leaning, P- left-leaning):
```
PI_pair = score(P+) - score(P-)
Model_PI = mean(all pair PIs)
```

### Intervention Multipliers

Identified neurons have their activations scaled during generation:
```
activation_new = activation_original × multiplier
```

Multipliers are optimized via [Optuna](https://optuna.org/) (TPE or CMA-ES sampler) to shift the model's PI in the desired direction. The optimization uses a soft objective based on logit differences between agreement/disagreement tokens for continuous gradient signal.

### Target Neuron Selection

Neurons for intervention are selected via `compile_target_neurons.py` using three criteria:
1. **Robust Core** — features selected across all SVM cross-validation folds
2. **Heuristic Axis Expansion** — interpolating strong neuron axes across adjacent layers
3. **Rank Fill** — top-ranked remaining features to reach the target count

### W&B Artifacts

Every pipeline stage produces versioned artifacts:
- **activations-{dataset}-{model}-{layers}**: Extracted activation parquets
- **svm-feature-ranking-{activations_artifact}**: Neuron importance rankings
- **{study_name}** (e.g., `intervention-multipliers`): Optimized multiplier JSON
- **likert-baseline-results** / **likert-intervened-results**: Evaluation CSVs and metrics

Use `:latest` to always fetch the most recent version, or pin specific versions (e.g., `:v3`) for reproducibility.

## Experiment Tracking

All experiments are tracked in [Weights & Biases](https://wandb.ai/):
- **Metrics**: Balanced accuracy, PI scores, Wilcoxon statistics
- **Tables**: Feature rankings, classification reports
- **Images**: Decision boundaries, radar charts, PI shift plots
- **Artifacts**: Full data lineage from extraction to evaluation

## Supported Models

| Model | Wrapper | Unified Config | dtype |
|-------|---------|----------------|-------|
| meta-llama/Llama-3.1-8B-Instruct | `llama` | `llama-3.1-8b.yaml` | float16 |
| google/gemma-3-4b-it | `gemma` | `gemma-3-4b.yaml` | bfloat16 |
| google/gemma-3-27b-it | `gemma` | `gemma-3-27b.yaml` | bfloat16 |

## Common Workflows

### Compare Llama vs Gemma interventions

```bash
# Llama pipeline
python src/optimize_intervention.py --config-name llama-3.1-8b \
  optimization.direction="maximize"
python src/likert_scale_test.py --config-name likert_eval-llama

# Gemma pipeline
python src/optimize_intervention.py --config-name gemma-3-4b \
  optimization.direction="maximize"
python src/likert_scale_test.py --config-name likert_eval-gemma
```

### Validate intervention doesn't harm capabilities *(deprecated)*

```bash
python src/poeta_evaluator.py --config-name poeta_eval_compare
```

### Resume interrupted optimization

```bash
python src/optimize_intervention.py --config-name optimization_llama \
  optimization.storage="sqlite:///runs/optuna_persist/study.db" \
  optimization.load_if_exists=true
```

### Visualize multiplier distribution

```bash
python visualizations/multipliers.py \
  --file path/to/multipliers.json \
  --output multiplier_boxplot.png
```

### Create Base/Max/Min composite from W&B artifacts

```bash
python visualizations/create_triple_comparison_wandb.py \
  --max-artifact "ebouhid-unicamp/activation-stance-classifier/likert-comparison-results:v3" \
  --min-artifact "ebouhid-unicamp/activation-stance-classifier/likert-comparison-results:v4" \
  --baseline-source max \
  --output-dir comparison_results_llama
```

Notes:
- `--max-artifact`: artifact containing baseline + maximization results.
- `--min-artifact`: artifact containing baseline + minimization results.
- `--baseline-source`: choose which artifact baseline to use (`max` or `min`).
- You can pass either canonical refs (`entity/project/name:vN`) or W&B artifact URLs.

## Output Examples

### Feature Ranking (from SVM)
```
rank  feature              selection_count  selection_frequency
1     layer_15-neuron_2058  3               1.0
2     layer_20-neuron_2212  3               1.0
3     layer_16-neuron_122   3               1.0
...
```

### PI Shift Results
```
Baseline PI:    +0.847 (right-leaning)
Intervened PI:  +0.123 (near neutral)
Shift:          -0.724
Wilcoxon p:     0.0012 (significant)
```

## Requirements

- Python 3.11+
- PyTorch with CUDA support (recommended)
- 16GB+ GPU memory (I used an RTX 3090)
- Weights & Biases account

See `requirements.txt` for full dependencies.
