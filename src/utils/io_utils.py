import os
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Union


def ensure_output_dirs(*dirs: Union[str, Path]) -> None:
    """Create directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def save_checkpoint(model: torch.nn.Module, filepath: Union[str, Path]) -> None:
    """Save model state dict to checkpoint file."""
    filepath = Path(filepath)
    directory = filepath.parent
    ensure_output_dirs(directory)
    torch.save(model.state_dict(), filepath)
    print(f"Checkpoint saved to {filepath}")


def save_run_results(filepath: Union[str, Path], **results_dict: Any) -> None:
    """
    Save training results to text file in existing format.

    Args:
        filepath: Path to save results (e.g., 'results/AccLoss2.txt')
        **results_dict: Dictionary of results to save (e.g., num_epochs=3000, batch_size=64)
    """
    filepath = Path(filepath)
    directory = filepath.parent
    ensure_output_dirs(directory)

    with open(filepath, 'a') as f:
        for key, value in results_dict.items():
            if isinstance(value, list):
                f.write(f"{key}= {value}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")

    print(f"Results saved to {filepath}")
