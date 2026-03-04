import pandas as pd
import torch
from tqdm import tqdm
from model_factory import get_model_wrapper
from activation_df import ActivationDataFrame
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import wandb


def get_last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Finds the index of the last non-padding token for each sequence in the batch.
    Assuming attention_mask is 1 for tokens and 0 for padding.
    """
    return attention_mask.sum(dim=1) - 1


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Configuration from Hydra
    batch_size = cfg.extraction.batch_size
    device = cfg.extraction.device if torch.cuda.is_available(
    ) and cfg.extraction.device == "cuda" else "cpu"

    # Get Hydra output directory
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir

    # W&B configuration
    wandb_cfg = cfg.get('wandb', {})

    # Resolve input path: prefer W&B artifact if provided
    dataset_artifact_name = cfg.data.get('dataset_artifact_name', None)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    if dataset_artifact_name:
        # Initialize W&B early to download artifact
        wandb.init(
            project=wandb_cfg.get('project', 'activation-bias-classifier'),
            name=wandb_cfg.get('run_name', None),
            job_type="extraction",
            config=wandb_config,
        )
        print(f"Downloading dataset artifact: {dataset_artifact_name}")
        artifact = wandb.use_artifact(dataset_artifact_name, type='dataset')
        artifact_dir = artifact.download()
        # Find CSV file in artifact
        csv_files = [f for f in os.listdir(artifact_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(
                f"No CSV file found in artifact {dataset_artifact_name}")
        input_path = os.path.join(artifact_dir, csv_files[0])
        print(f"Using dataset from artifact: {input_path}")
        wandb_initialized = True
    else:
        input_path = hydra.utils.to_absolute_path(cfg.data.input_csv)
        wandb_initialized = False

    # Determine artifact name based on dataset and model
    dataset_name = os.path.basename(input_path).replace(
        '.csv', '').replace('_propositions', '')
    model_short = cfg.model.wrapper  # 'llama' or 'gemma'
    layers = cfg.extraction.layers
    layers_str = 'all' if layers == 'all' else f"L{str(layers)}"
    artifact_name = f"activations-{dataset_name}-{model_short}-{layers_str}"
    output_path = f"data/{artifact_name}.parquet"

    # Initialize W&B with job_type="extraction" (if not already initialized for artifact download)
    if not wandb_initialized:
        wandb.init(
            project=wandb_cfg.get('project', 'activation-bias-classifier'),
            name=wandb_cfg.get('run_name', None),
            job_type="extraction",
        )

    # Update W&B config
    wandb.config.update({
        'input_csv': input_path,
        'dataset_artifact': dataset_artifact_name,
        'output_file': output_path,
        'batch_size': batch_size,
        'device': device,
        'layers': layers_str,
        'max_length': cfg.extraction.max_length,
        'model_name': cfg.model.name,
        'model_wrapper': cfg.model.wrapper,
    })

    # 1. Load Data
    print(f"Loading data from {input_path}...")
    # For demonstration, creating a dummy dataframe if file doesn't exist
    if not os.path.exists(input_path):
        print("Input file not found. Creating dummy data for demonstration.")
        df = pd.DataFrame({
            'statement': ["This is a test sentence.", "Another political statement.", "Short one."] * 10,
            'pol_label_human': ["neutral", "political", "neutral"] * 10
        })
    else:
        df = pd.read_csv(input_path)
        # Ensure columns exist
        if 'statement' not in df.columns or 'pol_label_human' not in df.columns:
            raise ValueError(
                "Input DataFrame must contain 'statement' and 'pol_label_human' columns.")

    print(f"Loaded {len(df)} samples.")

    # 2. Initialize Model using factory
    print(f"Initializing model...")
    wrapper = get_model_wrapper(cfg)
    if wrapper.model.tokenizer is None:
        raise ValueError("The model wrapper must have a tokenizer.")
    print(f"Loaded model: {wrapper.model.cfg.model_name}")

    # Ensure tokenizer has a pad token
    if wrapper.model.tokenizer.pad_token is None:
        wrapper.model.tokenizer.pad_token = wrapper.model.tokenizer.eos_token

    # Resolve layers list (handle 'all' option)
    layers_cfg = cfg.extraction.layers
    if isinstance(layers_cfg, str):
        layers = list(range(wrapper.n_layers))
    else:
        layers = list(layers_cfg)

    # 3. Initialize Accumulator with layer info
    d_model = wrapper.model.cfg.d_model
    activation_df = ActivationDataFrame(layers=layers, d_model=d_model)

    # 4. Processing Loop
    print("Starting extraction loop...")

    # Create batches
    total_samples = len(df)

    for start_idx in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_df = df.iloc[start_idx:end_idx]

        texts = batch_df['statement'].tolist()
        labels = batch_df['pol_label_human'].tolist()

        # Tokenize
        encoding = wrapper.model.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=cfg.extraction.max_length
        ).to(device)

        input_ids = torch.Tensor(encoding['input_ids'])
        attention_mask = torch.Tensor(encoding['attention_mask'])

        try:
            layer_activations = wrapper.get_layer_activations(
                input_ids, layers=cfg.extraction.layers).to(device)
        except Exception as e:
            print(f"Error processing batch {start_idx}-{end_idx}: {e}")
            continue

        padding_side = wrapper.model.tokenizer.padding_side

        batch_indices = torch.arange(input_ids.shape[0], device=device)

        if padding_side == 'left':
            last_token_indices = -1
            final_activations = layer_activations[:, -1, :]
        else:
            last_token_indices = attention_mask.sum(dim=1).to(device) - 1
            final_activations = layer_activations[batch_indices,
                                                  last_token_indices, :]

        # Add to accumulator
        activation_df.add_batch(final_activations, labels)

    # 5. Save Results
    print(f"Saving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    activation_df.save(output_path)
    full_output_path = os.path.join(os.getcwd(), output_path)
    print(f"Done. Saved to {full_output_path}")

    # --- ARTIFACT: Log activations as versioned dataset artifact ---

    activations_artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        description=f"Extracted activations from {cfg.model.name} on {dataset_name} dataset",
        metadata={
            'model_name': cfg.model.name,
            'model_wrapper': cfg.model.wrapper,
            'input_csv': input_path,
            'n_samples': total_samples,
            'n_layers': len(layers),
            'layers': layers,
            'd_model': d_model,
            'batch_size': batch_size,
            'max_length': cfg.extraction.max_length,
        }
    )
    activations_artifact.add_file(full_output_path)
    wandb.log_artifact(activations_artifact)
    print(f"Activations artifact logged: {artifact_name}")

    # Log summary metrics
    wandb.summary.update({
        'n_samples': total_samples,
        'n_layers': len(layers),
        'd_model': d_model,
        'n_features': len(layers) * d_model,
    })

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
