import wandb
import json
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sentence_transformers import SentenceTransformer, util


WANDB_ENTITY_PROJECT = "ebouhid-unicamp/activation-stance-classifier"
OUTPUT_DIR = "poeta_similarity_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

VERSION_MAP = {
    "v0": "gemma3-4B",
    "v1": "Llama3.1-8B",
    "v2": "Qwen3-8B",
    "v3": "Phi3-mini"
}


def fetch_and_load_artifact(run, artifact_path):
    """Downloads the wandb artifact and loads the _outputs.json file."""
    print(f"Fetching artifact: {artifact_path}")
    # Adjust type if different
    artifact = run.use_artifact(artifact_path)
    artifact_dir = artifact.download()

    # Find the JSON file ending in _outputs.json
    json_files = glob.glob(os.path.join(artifact_dir, "*_outputs.json"))
    if not json_files:
        raise FileNotFoundError(f"No *_outputs.json found in {artifact_dir}")

    with open(json_files[0], 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_mc_answer(ans):
    """Cleans multiple choice answers by stripping whitespace, converting to uppercase, and removing periods."""
    return str(ans).strip().upper().replace('.', '')


def plot_transition_heatmap(baseline_dict, intervened_dict, task_name, model_name, mapped_model_name):
    """Calculates Cohen's Kappa and plots a transition heatmap for MC tasks."""
    # Extract just the string answers, assuming 'dynamic-random' holds the data
    keys = sorted(baseline_dict['dynamic-random'].keys(), key=int)
    y_base = [clean_mc_answer(
        baseline_dict['dynamic-random'][k][0]) for k in keys]
    y_interv = [clean_mc_answer(
        intervened_dict['dynamic-random'][k][0]) for k in keys]

    # Calculate Cohen's Kappa for statistical agreement
    kappa = cohen_kappa_score(y_base, y_interv)

    # Create Confusion Matrix (Transition Matrix)
    labels = sorted(list(set(y_base + y_interv)))
    cm = confusion_matrix(y_base, y_interv, labels=labels)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(
        f"Transition Heatmap: Baseline vs {model_name} ({mapped_model_name})\nTask: {task_name} | Cohen's Kappa: {kappa:.3f}")
    plt.xlabel(f"{model_name} Model Answers")
    plt.ylabel("Baseline Answers")
    plt.tight_layout()
    output_filename = f"heatmap_{task_name}_{model_name}_{mapped_model_name}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=300)
    plt.close()
    print(
        f"Saved categorical heatmap for {task_name} - {model_name} ({mapped_model_name}). Kappa: {kappa:.3f}")

    return kappa


def plot_pairwise_similarity_heatmap(baseline_dict, min_dict, max_dict, task_name, mapped_model_name):
    """Calculates median semantic similarity and plots a 3x3 pairwise heatmap."""
    print(
        f"Loading SentenceTransformer model for semantic similarity (Model: {mapped_model_name})...")
    # Adjust to your preferred embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    keys = sorted(baseline_dict['dynamic-random'].keys(), key=int)

    # Extract texts
    texts = {
        "Baseline": [baseline_dict['dynamic-random'][k][0] for k in keys],
        "Min": [min_dict['dynamic-random'][k][0] for k in keys],
        "Max": [max_dict['dynamic-random'][k][0] for k in keys]
    }

    # Encode all texts into embeddings
    embeddings = {name: model.encode(
        text_list, convert_to_tensor=True) for name, text_list in texts.items()}

    # Compute 3x3 similarity matrix
    model_names = ["Baseline", "Min", "Max"]
    sim_matrix = np.zeros((3, 3))

    for i, model_a in enumerate(model_names):
        for j, model_b in enumerate(model_names):
            # Compute pairwise cosine similarity for all items
            cosine_scores = util.cos_sim(
                embeddings[model_a], embeddings[model_b])
            # Extract the diagonal (similarity of corresponding paired answers)
            paired_sims = np.diag(cosine_scores.cpu().numpy())
            # Use Median to avoid skewness from extreme outliers
            sim_matrix[i, j] = np.median(paired_sims)

    # Save the similarity matrix to CSV (one file per model)
    sim_df = pd.DataFrame(sim_matrix, index=model_names, columns=model_names)
    matrix_csv_filename = f"similarity_matrix_{task_name}_{mapped_model_name}.csv"
    matrix_csv_path = os.path.join(OUTPUT_DIR, matrix_csv_filename)
    sim_df.to_csv(matrix_csv_path)
    print(
        f"Saved similarity matrix CSV for {task_name} ({mapped_model_name}) to {matrix_csv_path}")

    # Plotting
    plt.figure(figsize=(7, 5))
    sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=model_names, yticklabels=model_names, vmin=0.0, vmax=1.0)
    plt.title(
        f"Median Pairwise Semantic Similarity\nTask: {task_name} | Model: {mapped_model_name}")
    plt.tight_layout()
    output_filename = f"similarity_heatmap_{task_name}_{mapped_model_name}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=300)
    plt.close()
    print(f"Saved similarity heatmap for {task_name} ({mapped_model_name}).")


def main():
    # Initialize wandb run (needed to download artifacts)
    run = wandb.init(project="visualization_pipeline", job_type="analysis")

    kappa_results = []

    for version_tag, mapped_name in VERSION_MAP.items():
        print(f"--- Processing {mapped_name} ({version_tag}) ---")

        artifact_paths = {
            "baseline": f"{WANDB_ENTITY_PROJECT}/poeta-baseline-baseline-results:{version_tag}",
            "min": f"{WANDB_ENTITY_PROJECT}/poeta-minimize-intervened-results:{version_tag}",
            "max": f"{WANDB_ENTITY_PROJECT}/poeta-maximize-intervened-results:{version_tag}"
        }

        # Load data
        data = {}
        try:
            for condition, path in artifact_paths.items():
                data[condition] = fetch_and_load_artifact(run, path)
        except Exception as e:
            print(f"Skipping {version_tag} ({mapped_name}) due to error: {e}")
            continue

        # Process Multiple Choice Tasks
        mc_tasks = ['enem_2022_greedy', 'math_mc_greedy']
        for task in mc_tasks:
            if task in data['baseline']:
                kappa_min = plot_transition_heatmap(
                    data['baseline'][task], data['min'][task], task, "Min", mapped_name)
                kappa_max = plot_transition_heatmap(
                    data['baseline'][task], data['max'][task], task, "Max", mapped_name)

                kappa_results.append({
                    "Model": mapped_name,
                    "Task": task,
                    "Intervention": "Min",
                    "Kappa": kappa_min
                })
                kappa_results.append({
                    "Model": mapped_name,
                    "Task": task,
                    "Intervention": "Max",
                    "Kappa": kappa_max
                })

        # Process Text Generation Task
        if 'faquad' in data['baseline']:
            plot_pairwise_similarity_heatmap(
                data['baseline']['faquad'],
                data['min']['faquad'],
                data['max']['faquad'],
                "faquad",
                mapped_name
            )

    wandb.finish()

    if kappa_results:
        df_kappa = pd.DataFrame(kappa_results)
        csv_path = os.path.join(OUTPUT_DIR, "kappa_coefficients.csv")
        df_kappa.to_csv(csv_path, index=False)
        print(f"Saved kappa coefficients to {csv_path}")


if __name__ == "__main__":
    main()
