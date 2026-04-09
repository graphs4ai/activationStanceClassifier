from model_factory import get_wrapper_class
import os
import sys
import json
import glob
import torch
import logging
import wandb
from datetime import datetime
from typing import Optional, Dict, List, Any, Iterable, Tuple
from pathlib import Path
from contextlib import contextmanager

import hydra
from omegaconf import DictConfig, OmegaConf


class TeeLogger:
    """Logger that writes to both console and file simultaneously."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.file = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


@contextmanager
def tee_output(log_file: Path):
    """Context manager to tee stdout/stderr to a log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    tee_stdout = TeeLogger(log_file)
    tee_stderr = TeeLogger(log_file.with_suffix('.err.log'))

    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    try:
        yield log_file
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tee_stdout.close()
        tee_stderr.close()


# Add PoETa to path (sibling directory) - must be added before lm_eval imports
# This ensures we use the local fork with fixes applied
POETA_PATH = Path(__file__).parent.parent.parent / "PoETaV2"
PROJECT_PATH = Path(__file__).parent.parent

if POETA_PATH.exists():
    sys.path.insert(0, str(POETA_PATH))

# Change to PoETa directory for conversation.py import
_original_cwd = os.getcwd()
os.chdir(POETA_PATH)

try:
    from lm_eval.base import BaseLM
    from lm_eval import tasks, evaluator
    from lm_eval.utils import stop_sequences_criteria
finally:
    os.chdir(_original_cwd)


class IntervenedLlamaLM(BaseLM):
    """
    PoETa-compatible Language Model wrapper for supported TransformerLens models
    (Llama 3.1 / Gemma 3 / Qwen 3 / Phi-4) with activation interventions.

    This class bridges the model wrappers (which use TransformerLens/HookedTransformer)
    with the PoETa evaluation framework, enabling evaluation of models with modified
    internal activations.
    """

    def __init__(
        self,
        device: str = "cuda",
        pretrained: str = "meta-llama/Llama-3.1-8B-Instruct",
        batch_size: int = 1,
        activation_multipliers: Optional[Dict[str, float]] = None,
        wrapper_type: str = "llama",
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the intervened model for PoETa evaluation.

        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            pretrained: HuggingFace model identifier
            batch_size: Batch size for evaluation
            activation_multipliers: Dict mapping 'layer_X-neuron_Y' to multiplier values
                                   for activation interventions. None for baseline.
            wrapper_type: "llama", "gemma", "qwen", or "phi"
            dtype: Model data type (e.g. torch.bfloat16). Used if wrapper_type needs it.
        """
        super().__init__()

        self._device = torch.device(device)
        self.batch_size_per_gpu = batch_size
        self.activation_multipliers = activation_multipliers or {}

        # Initialize wrapper using factory
        print(f"Loading model: {pretrained} (wrapper: {wrapper_type})")
        print(
            f"Activation interventions: {len(self.activation_multipliers)} neurons")
        WrapperClass = get_wrapper_class(wrapper_type)
        if dtype is not None:
            self.wrapper = WrapperClass(
                model_name=pretrained, device=device, dtype=dtype)
        else:
            self.wrapper = WrapperClass(model_name=pretrained, device=device)

        # Get references to model and tokenizer from wrapper
        self.model = self.wrapper.model
        self.tokenizer = self.model.tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        # Parse intervention layers for hook setup
        self._intervention_layers = self._parse_intervention_layers()

    def _parse_intervention_layers(self) -> Dict[int, Dict[int, float]]:
        """Parse activation_multipliers into per-layer neuron dictionaries."""
        layer_neuron_multipliers = {}
        for feature_name, multiplier in self.activation_multipliers.items():
            parts = feature_name.split('-')
            layer_idx = int(parts[0].split('_')[1])
            neuron_idx = int(parts[1].split('_')[1])
            if layer_idx not in layer_neuron_multipliers:
                layer_neuron_multipliers[layer_idx] = {}
            layer_neuron_multipliers[layer_idx][neuron_idx] = multiplier
        return layer_neuron_multipliers

    def _make_intervention_hook(self, neuron_multipliers: Dict[int, float]):
        """Create a hook function that applies neuron-specific multipliers."""
        def hook(resid_pre: torch.Tensor, hook):
            modified = resid_pre.clone()
            for neuron_idx, multiplier in neuron_multipliers.items():
                modified[:, :, neuron_idx] = modified[:,
                                                      :, neuron_idx] * multiplier
            return modified
        return hook

    def _get_intervention_hooks(self) -> List[tuple]:
        """Build list of (hook_point, hook_fn) tuples for interventions."""
        fwd_hooks = []
        for layer_idx, neuron_mults in self._intervention_layers.items():
            hook_point = f"blocks.{layer_idx}.hook_resid_pre"
            fwd_hooks.append(
                (hook_point, self._make_intervention_hook(neuron_mults)))
        return fwd_hooks

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        """Maximum context length the model can handle."""
        try:
            # Prefer tokenizer's model_max_length as it's more accurate for Llama 3.1
            if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length < 1e10:
                return self.tokenizer.model_max_length
            # Fallback to TransformerLens cfg.n_ctx
            if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'n_ctx'):
                return self.model.cfg.n_ctx
            return 131072  # Llama 3.1 default context
        except Exception:
            return 131072

    @property
    def max_gen_toks(self):
        return 2048

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional activation interventions.

        Args:
            inps: Input token IDs [batch, sequence]

        Returns:
            Logits tensor [batch, sequence, vocab_size]
        """
        with torch.no_grad():
            if self._intervention_layers:
                # Run with intervention hooks
                fwd_hooks = self._get_intervention_hooks()
                logits = self.model.run_with_hooks(
                    inps.to(self._device),
                    fwd_hooks=fwd_hooks
                )
            else:
                # Standard forward pass (baseline)
                logits = self.model(inps.to(self._device))

        return logits[:, :, :self.vocab_size]

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop_sequences: List[str]
    ) -> torch.Tensor:
        """
        Generate text with optional activation interventions.

        Args:
            context: Input token IDs [batch, seq_len]
            max_length: Maximum total length (input + generated)
            stop_sequences: List of strings to stop generation at

        Returns:
            Generated token IDs including input
        """
        max_new_tokens = max_length - context.shape[1]

        # Build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop_sequences, context.shape[1], context.shape[0]
        )

        with torch.no_grad():
            if self._intervention_layers:
                # Add hooks temporarily for generation
                fwd_hooks = self._get_intervention_hooks()
                for hook_point, hook_fn in fwd_hooks:
                    self.model.add_hook(hook_point, hook_fn)

                try:
                    output = self.model.generate(
                        context.to(self._device),
                        max_new_tokens=max_new_tokens,
                        stop_at_eos=True,
                        eos_token_id=self.eot_token_id,
                        do_sample=False,
                        verbose=False
                    )
                finally:
                    self.model.reset_hooks()
            else:
                # Standard generation (baseline)
                output = self.model.generate(
                    context.to(self._device),
                    max_new_tokens=max_new_tokens,
                    stop_at_eos=True,
                    eos_token_id=self.eot_token_id,
                    do_sample=False,
                    verbose=False
                )

        return output


def run_poeta_evaluation(
    activation_multipliers: Optional[Dict[str, float]] = None,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    wrapper_type: str = "llama",
    tasks_list: Optional[List[str]] = None,
    num_fewshot: int = 0,
    prompt_modes: str = "dynamic-random",
    batch_size: int = 1,
    device: str = "cuda",
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    description_dict_path: Optional[str] = None,
    log_to_wandb: bool = False,
    wandb_prefix: str = "",
    dtype: Optional[torch.dtype] = None,
    evaluation_variant: str = "baseline",
    multiplier_source: str = "none",
    multiplier_artifact_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run PoETa V2 benchmark evaluation on a model with optional interventions.

    Args:
        activation_multipliers: Dict of 'layer_X-neuron_Y' -> multiplier for interventions.
                               Use None or {} for baseline evaluation.
        model_name: HuggingFace model identifier
        wrapper_type: "llama", "gemma", "qwen", or "phi"
        tasks_list: List of PoETa task names to run. None runs all tasks.
        num_fewshot: Number of few-shot examples
        prompt_modes: Prompt mode(s), comma-separated
        batch_size: Evaluation batch size
        device: Device to run on ('cuda' or 'cpu')
        limit: Limit number of examples per task (for testing)
        output_path: Path to save results JSON
        description_dict_path: Path to task description JSON
        log_to_wandb: Whether to log results to wandb
        wandb_prefix: Prefix for wandb metric names

    Returns:
        Dictionary containing evaluation results
    """
    # Default to all tasks if none specified
    if tasks_list is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = tasks_list

    # Parse prompt modes
    prompt_modes_list = prompt_modes.split(",")

    # Load description dict if provided
    description_dict = {}
    if description_dict_path:
        desc_path = Path(description_dict_path)
        if not desc_path.is_absolute():
            desc_path = POETA_PATH / description_dict_path
        if desc_path.exists():
            with open(desc_path, 'r') as f:
                description_dict = json.load(f)

    # Create our custom model
    model = IntervenedLlamaLM(
        device=device,
        pretrained=model_name,
        batch_size=batch_size,
        activation_multipliers=activation_multipliers,
        wrapper_type=wrapper_type,
        dtype=dtype,
    )

    # Determine output directory (convert to absolute path for saving outside PoETa dir)
    output_dir = None
    abs_output_path = None
    if output_path:
        abs_output_path = Path(output_path)
        if not abs_output_path.is_absolute():
            abs_output_path = PROJECT_PATH / output_path
        abs_output_path = abs_output_path.resolve()
        output_dir = str(abs_output_path.parent)
        if output_dir == "":
            output_dir = "."

    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Running PoETa V2 Evaluation")
    print(f"Model: {model_name}")
    print(f"Interventions: {len(activation_multipliers or {})} neurons")
    print(f"Tasks: {len(task_names)} tasks")
    print(f"Limit: {limit}")
    print(f"{'='*60}\n")

    # Change to PoETa directory for task data loading
    original_cwd = os.getcwd()
    os.chdir(POETA_PATH)

    try:
        results = evaluator.simple_evaluate(
            model=model,
            model_args="",  # Not used since we pass model instance
            tasks=task_names,
            num_fewshot=num_fewshot,
            prompt_modes=prompt_modes_list,
            batch_size=batch_size,
            device=device,
            no_cache=True,
            limit=limit,
            description_dict=description_dict,
            conversation_template=None,
            prompt_as_single_user_message=False,
            check_integrity=False,
            output_dir=output_dir,
        )
    finally:
        os.chdir(original_cwd)

    # Add metadata to results
    results['metadata'] = {
        'model_name': model_name,
        'activation_multipliers': activation_multipliers,
        'num_interventions': len(activation_multipliers or {}),
        'evaluation_type': 'intervened' if activation_multipliers else 'baseline',
        'evaluation_variant': evaluation_variant,
        'multiplier_source': multiplier_source,
        'multiplier_artifact_name': multiplier_artifact_name,
        'timestamp': datetime.now().isoformat(),
        'tasks': task_names,
        'num_fewshot': num_fewshot,
        'limit': limit,
    }

    def make_serializable(obj):
        """Recursively convert non-serializable objects to strings."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert non-serializable objects to their string representation
            return str(obj)

    serializable_results = make_serializable(results)

    # Save results
    if abs_output_path:
        abs_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(abs_output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to: {abs_output_path}")

        if 'outputs' in serializable_results:
            outputs_path = abs_output_path.parent / \
                f"{abs_output_path.stem}_outputs.json"
            with open(outputs_path, 'w') as f:
                json.dump(serializable_results['outputs'], f, indent=2)
            print(f"Outputs saved to: {outputs_path}")

    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(evaluator.make_table(results))

    # Log to wandb if requested
    if log_to_wandb and wandb.run is not None:
        wandb_metrics = {}
        table_rows = []

        # Log evaluation results
        for task_name, task_metrics in results.get('results', {}).items():
            if isinstance(task_metrics, dict):
                for key, value in task_metrics.items():
                    # Handle nested prompt_mode structure (e.g. "dynamic-random": {"acc": 1.0})
                    if isinstance(value, dict):
                        for metric, val in value.items():
                            if isinstance(val, (int, float)):
                                wandb_metrics[f"{wandb_prefix}{task_name}/{metric}"] = val
                                table_rows.append(
                                    [task_name, key, metric, val])
                    # Handle direct metric structure
                    elif isinstance(value, (int, float)):
                        wandb_metrics[f"{wandb_prefix}{task_name}/{key}"] = value
                        table_rows.append([task_name, "default", key, value])

        # Create and add table
        if table_rows:
            table = wandb.Table(
                columns=["Task", "Prompt Mode", "Metric", "Value"], data=table_rows)
            wandb_metrics[f"{wandb_prefix}results_table"] = table

        # Log outputs table
        if 'outputs' in results:
            output_rows = []
            for task_name, task_outputs in results['outputs'].items():
                for prompt_mode, doc_outputs in task_outputs.items():
                    for doc_id, preds in doc_outputs.items():
                        pred_str = str(preds)
                        if isinstance(preds, list) and len(preds) == 1:
                            pred_str = str(preds[0])
                        output_rows.append(
                            [task_name, prompt_mode, str(doc_id), pred_str])

            if output_rows:
                out_table = wandb.Table(
                    columns=["Task", "Prompt Mode", "Doc ID", "Prediction"], data=output_rows)
                wandb_metrics[f"{wandb_prefix}model_outputs"] = out_table

        wandb.log(wandb_metrics)

    return serializable_results


def _load_multipliers_from_config(cfg: DictConfig) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    """
    Load activation multipliers using the same logic as likert_scale_test.py.

    Resolution order:
      1. likert.multiplier_artifact_name (W&B artifact)
      2. likert.activation_multipliers (inline config)
      3. activation_multipliers (top-level config, legacy)

    Args:
        cfg: Hydra DictConfig

    Returns:
        Tuple:
          - activation_multipliers dict or None
          - provenance metadata dict with source/artifact_name/variant
    """
    likert_cfg = cfg.get('likert', {})
    if likert_cfg is None:
        likert_cfg = {}

    evaluation_variant = str(cfg.get('evaluation_variant', '') or '').strip().lower()
    if evaluation_variant not in {"baseline", "maximize", "minimize"}:
        evaluation_variant = "baseline"

    multiplier_artifact_name = likert_cfg.get('multiplier_artifact_name', None)

    provenance = {
        "source": "none",
        "artifact_name": multiplier_artifact_name,
        "variant": evaluation_variant,
    }

    if evaluation_variant == "baseline":
        return None, provenance

    # --- Option 1: Load from W&B artifact ---
    if multiplier_artifact_name:
        print(f"\nFetching multiplier artifact: {multiplier_artifact_name}")
        artifact = wandb.use_artifact(multiplier_artifact_name)
        artifact_dir = artifact.download()

        # Find optimization results JSON file
        json_files = glob.glob(os.path.join(
            artifact_dir, "optimization_results_*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No optimization_results_*.json found in artifact: {artifact_dir}")

        results_path = json_files[0]
        with open(results_path, 'r', encoding='utf-8') as f:
            opt_results = json.load(f)

        # Extract multipliers from best trial
        best_trial = opt_results.get('best_trial', {})
        activation_multipliers = best_trial.get('multipliers', {})
        print(
            f"Loaded {len(activation_multipliers)} multipliers from artifact")
        print(
            f"Artifact best trial value: {best_trial.get('soft_score', 'N/A')}")
        provenance["source"] = "artifact"
        return activation_multipliers, provenance

    # --- Option 2: Load from likert.activation_multipliers ---
    likert_multipliers = likert_cfg.get('activation_multipliers', None)
    if likert_multipliers is not None:
        activation_multipliers = {str(k): float(v)
                                  for k, v in dict(likert_multipliers).items()}
        provenance["source"] = "likert.activation_multipliers"
        return activation_multipliers, provenance

    # --- Option 3: Load from top-level activation_multipliers (legacy) ---
    if cfg.get('activation_multipliers'):
        activation_multipliers = OmegaConf.to_container(
            cfg.activation_multipliers)
        provenance["source"] = "activation_multipliers"
        return activation_multipliers, provenance

    return None, provenance


# Hydra configuration for running from command line
@hydra.main(config_path="../config", config_name="poeta_eval", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for Hydra-based PoETa evaluation.

    Supports unified config files (e.g. gemma-3-4b.yaml) where intervention
    multipliers are loaded from likert.multiplier_artifact_name or
    likert.activation_multipliers, following the same logic as likert_scale_test.py.

    Config should contain:
        - model / model_name: HuggingFace model path
        - poeta.tasks: List of task names or 'all'
        - poeta.num_fewshot: Number of few-shot examples
        - poeta.limit: Example limit (null for full evaluation)
        - poeta.output_dir: Where to save results
        - poeta.batch_size / poeta.device / poeta.prompt_modes
        - likert.multiplier_artifact_name: W&B artifact with optimized multipliers
        - likert.activation_multipliers: Inline multipliers dict
        - activation_multipliers: Legacy inline multipliers dict
    """
    # Change back to project directory (Hydra changes cwd to outputs/)
    os.chdir(PROJECT_PATH)

    print(OmegaConf.to_yaml(cfg))

    wandb_cfg = cfg.get('wandb', {})
    if wandb_cfg is None:
        wandb_cfg = {}

    log_to_wandb = wandb_cfg.get('log_to_wandb', False)

    # Get model configuration (support both old and new config format)
    model_cfg = cfg.get('model', {})
    model_name = model_cfg.get('name', cfg.get(
        'model_name', 'meta-llama/Llama-3.1-8B-Instruct'))
    wrapper_type = model_cfg.get('wrapper', 'llama')

    dtype_str = model_cfg.get('dtype', 'float16').lower()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    evaluation_variant = str(cfg.get('evaluation_variant', '') or '').strip().lower()
    if not evaluation_variant:
        evaluation_variant = "baseline"
    if evaluation_variant not in {"baseline", "maximize", "minimize"}:
        raise ValueError(
            f"Invalid evaluation_variant='{evaluation_variant}'. Expected one of: baseline, maximize, minimize."
        )

    # Prepare wandb config metadata
    likert_cfg = cfg.get('likert', {}) or {}
    multiplier_artifact_name = likert_cfg.get('multiplier_artifact_name', None)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config.update({
        'evaluation_variant': evaluation_variant,
        'multiplier_source': 'artifact' if multiplier_artifact_name else 'config_or_none',
        'multiplier_artifact_name': multiplier_artifact_name,
    })

    if log_to_wandb:
        wandb.init(
            project=wandb_cfg.get('project', 'activation-stance-classifier'),
            entity=wandb_cfg.get('entity', None),
            name=wandb_cfg.get('run_name', None),
            job_type="poeta_eval",
            tags=list(wandb_cfg.get('tags', [])),
            config=wandb_config,
        )
    else:
        # Initialize wandb in offline/disabled mode so wandb.use_artifact still works
        # when loading multipliers from artifacts
        if multiplier_artifact_name:
            wandb.init(
                project=wandb_cfg.get(
                    'project', 'activation-stance-classifier'),
                entity=wandb_cfg.get('entity', None),
                name=wandb_cfg.get('run_name', None),
                job_type="poeta_eval",
                config=wandb_config,
            )
            # Override: we need wandb for artifact loading, so enable logging
            log_to_wandb = True

    # Load activation multipliers (same logic as likert_scale_test.py)
    activation_multipliers, multiplier_provenance = _load_multipliers_from_config(
        cfg)
    multiplier_source = multiplier_provenance["source"]

    if evaluation_variant in {"maximize", "minimize"} and not activation_multipliers:
        raise ValueError(
            f"evaluation_variant='{evaluation_variant}' requires multipliers, but none were found."
        )
    if evaluation_variant == "baseline" and activation_multipliers:
        print(
            "\nBaseline variant requested: ignoring configured multipliers and running without intervention."
        )
        activation_multipliers = None
        multiplier_source = "none"
        multiplier_provenance["source"] = "none"

    if activation_multipliers:
        print(
            f"\nActivation intervention configured: {len(activation_multipliers)} neurons")
        print(f"Multiplier source: {multiplier_source}")
    else:
        print("\nNo activation interventions configured (baseline mode)")

    # Parse nested PoETa configuration
    poeta_cfg = cfg.get('poeta', None)
    if not poeta_cfg:
        raise ValueError(
            "Missing required 'poeta' section in config. "
            "All PoETa evaluation settings must be nested under 'poeta:'."
        )

    # Parse tasks
    tasks_list = None
    poeta_tasks = poeta_cfg.get('tasks')
    if poeta_tasks and poeta_tasks != 'all':
        if isinstance(poeta_tasks, str):
            tasks_list = poeta_tasks.split(',')
        else:
            # It's already a list (ListConfig)
            tasks_list = list(poeta_tasks)

    poeta_num_fewshot = int(poeta_cfg.get('num_fewshot', 0))
    poeta_limit = poeta_cfg.get('limit', None)
    poeta_prompt_modes = poeta_cfg.get('prompt_modes', 'dynamic-random')
    poeta_device = poeta_cfg.get('device', 'cuda')
    poeta_batch_size = int(poeta_cfg.get('batch_size', 1))
    poeta_output_dir = poeta_cfg.get('output_dir', 'runs')
    poeta_save_logs = bool(poeta_cfg.get('save_logs', True))
    poeta_description_dict_path = poeta_cfg.get(
        'description_dict_path', 'description.json')
    poeta_compare_baseline = bool(poeta_cfg.get('compare_baseline', False))

    # Run single evaluation - follow convention: runs/date/poeta_time/
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    eval_type = 'intervened' if activation_multipliers else 'baseline'
    run_dir = Path(poeta_output_dir) / date_str / f"poeta_{time_str}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(run_dir / f"{evaluation_variant}_{eval_type}_results.json")
    log_file = run_dir / "evaluation.log"

    def run_single_eval():
        return run_poeta_evaluation(
            activation_multipliers=activation_multipliers,
            model_name=model_name,
            wrapper_type=wrapper_type,
            tasks_list=tasks_list,
            num_fewshot=poeta_num_fewshot,
            prompt_modes=poeta_prompt_modes,
            batch_size=poeta_batch_size,
            device=poeta_device,
            limit=poeta_limit,
            output_path=output_path,
            description_dict_path=poeta_description_dict_path,
            log_to_wandb=log_to_wandb,
            wandb_prefix=f"variant/{evaluation_variant}/",
            dtype=dtype,
            evaluation_variant=evaluation_variant,
            multiplier_source=multiplier_source,
            multiplier_artifact_name=multiplier_artifact_name,
        )

    if poeta_save_logs:
        with tee_output(log_file):
            print(f"Log file: {log_file}")
            print(f"Run directory: {run_dir}")
            print(f"Timestamp: {datetime.now().isoformat()}")
            print("\n" + "="*60)
            results = run_single_eval()
    else:
        results = run_single_eval()

    # Save config used for this run
    config_path = run_dir / "config.json"
    config_data = {
        'model_name': model_name,
        'evaluation_variant': evaluation_variant,
        'activation_multipliers': activation_multipliers,
        'multiplier_source': multiplier_source,
        'multiplier_artifact_name': multiplier_artifact_name,
        'multiplier_provenance': multiplier_provenance,
        'tasks': tasks_list,
        'num_fewshot': poeta_num_fewshot,
        'limit': poeta_limit,
        'prompt_modes': poeta_prompt_modes,
        'device': poeta_device,
        'batch_size': poeta_batch_size,
        'compare_baseline': poeta_compare_baseline,
        'output_dir': poeta_output_dir,
        'description_dict_path': poeta_description_dict_path,
        'timestamp': datetime.now().isoformat(),
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    # Log single-eval artifact to W&B
    if log_to_wandb and wandb.run is not None:
        _log_single_eval_artifact(
            run_dir=str(run_dir),
            results=results,
            eval_type=eval_type,
            evaluation_variant=evaluation_variant,
            model_name=model_name,
            multiplier_source=multiplier_source,
            multiplier_artifact_name=multiplier_artifact_name,
            n_multipliers=len(activation_multipliers or {}),
        )

    # Finish W&B run
    if wandb.run is not None:
        wandb.finish()

    return results


def _log_single_eval_artifact(
    run_dir: str,
    results: Dict[str, Any],
    eval_type: str,
    evaluation_variant: str,
    model_name: str,
    multiplier_source: str,
    multiplier_artifact_name: Optional[str],
    n_multipliers: int,
):
    """Log a single PoETa evaluation run as a W&B artifact."""
    artifact_name = f"poeta-{evaluation_variant}-{eval_type}-results"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="poeta-evaluation",
        description=f"PoETa V2 evaluation results ({evaluation_variant}/{eval_type})",
        metadata={
            'model_name': model_name,
            'eval_type': eval_type,
            'evaluation_variant': evaluation_variant,
            'multiplier_source': multiplier_source,
            'multiplier_artifact_name': multiplier_artifact_name,
            'n_multipliers': n_multipliers,
        }
    )

    # Add all JSON files in the run directory
    run_path = Path(run_dir)
    for json_file in run_path.glob("*.json"):
        artifact.add_file(str(json_file))
    for log_file in run_path.glob("*.log"):
        artifact.add_file(str(log_file))

    wandb.log_artifact(artifact)
    print(f"\nPoETa artifact logged to W&B: {artifact_name}")

    # Log summary metrics
    wandb.summary.update({
        'eval_type': eval_type,
        'evaluation_variant': evaluation_variant,
        'model_name': model_name,
        'n_multipliers': n_multipliers,
        'multiplier_source': multiplier_source,
    })


if __name__ == "__main__":
    main()
