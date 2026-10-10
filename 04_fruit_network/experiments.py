import json
import torch

from data import get_loaders, IMAGE_SIZE
from train import train
import model

SEED = 20
EPOCHS = 20
CONFIGS = [
    {"normalize": False, "lr": 0.01},
    {"normalize": False, "lr": 0.05},
    {"normalize": True, "lr": 0.01},
    {"normalize": True, "lr": 0.05},
]


def run_experiment(normalize: bool, lr: float, epochs: int = EPOCHS) -> list:
    """Trains a fresh network with these settings and returns its epoch losses."""
    # seeding inside the function: every run starts from the same weights
    # and sees the same shuffle order, so only the settings differ
    torch.manual_seed(SEED)
    train_loader, _, classes = get_loaders(normalize=normalize)
    net = model.FruitNetwork(IMAGE_SIZE * IMAGE_SIZE, len(classes))

    return train(net, train_loader, epochs=epochs, lr=lr)


def print_summary(results: dict) -> None:
    """Prints the results as a Markdown table, ready to paste into the README."""
    print("\n| Run | Loss at epoch 1 | Loss at last epoch | Lowest loss (epoch) |")
    print("|-----|-----------------|--------------------|---------------------|")
    for label, losses in results.items():
        lowest = min(losses)
        print(f"| {label} | {losses[0]:.4f} | {losses[-1]:.4f} | {lowest:.4f} ({losses.index(lowest) + 1}) |")


if __name__ == "__main__":
    results = {}

    for config in CONFIGS:
        normalize = config["normalize"]
        lr = config["lr"]

        print(f"Running experiment: normalize={normalize}, lr={lr}")
        results[f"norm={normalize}, lr={lr}"] = run_experiment(normalize, lr)

    with open("experiments.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Saved all experiment results to experiments.json")
    print_summary(results)