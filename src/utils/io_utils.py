import os
import torch


def ensure_output_dirs(*dirs):
    """Create directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def save_checkpoint(model, filepath):
    """Save model state dict to checkpoint file."""
    directory = os.path.dirname(filepath)
    ensure_output_dirs(directory)
    torch.save(model.state_dict(), filepath)
    print(f"Checkpoint saved to {filepath}")


def save_run_results(filepath, **results_dict):
    """
    Save training results to text file in existing format.

    Args:
        filepath: Path to save results (e.g., 'results/AccLoss2.txt')
        **results_dict: Dictionary of results to save (e.g., num_epochs=3000, batch_size=64)
    """
    directory = os.path.dirname(filepath)
    ensure_output_dirs(directory)

    with open(filepath, 'a') as f:
        for key, value in results_dict.items():
            if isinstance(value, list):
                f.write(f"{key}= {value}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")

    print(f"Results saved to {filepath}")
