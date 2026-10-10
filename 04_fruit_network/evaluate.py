import json
from collections import Counter

import torch

from data import get_loaders, IMAGE_SIZE
from train import CHECKPOINT
import model


def evaluate(net, loader, classes: list) -> dict:
    """
    Measures how often net predicts the right class.
    returns: {"accuracy": float,
              "per_class": {class name: accuracy},
              "confusions": {(true class, predicted class): count}}
    """
    net.eval()
    correct = Counter()
    total = Counter()
    confusions = Counter()

    with torch.no_grad():
        for images, labels in loader:
            # the predicted class is the one with the largest raw score
            predictions = net(images).argmax(dim=1)

            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                total[label] += 1
                if prediction == label:
                    correct[label] += 1
                else:
                    confusions[(classes[label], classes[prediction])] += 1

    return {
        "accuracy": sum(correct.values()) / sum(total.values()),
        "per_class": {classes[i]: correct[i] / total[i] for i in sorted(total)},
        "confusions": dict(confusions),
    }


def count_images(loader, classes: list) -> dict:
    """Number of images per class in the loader's dataset."""
    counts = Counter(loader.dataset.targets)
    return {classes[i]: counts[i] for i in sorted(counts)}


def load_network(path: str = CHECKPOINT) -> tuple:
    """
    returns: (network with the saved weights, the checkpoint dictionary)
    """
    checkpoint = torch.load(path, weights_only=True)
    if checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError(
            f"{path} was trained on {checkpoint['image_size']}x{checkpoint['image_size']} "
            f"images but data.py now uses {IMAGE_SIZE}x{IMAGE_SIZE}"
        )
    net = model.FruitNetwork(IMAGE_SIZE * IMAGE_SIZE, len(checkpoint["classes"]))
    net.load_state_dict(checkpoint["state_dict"])
    return net, checkpoint


if __name__ == "__main__":
    net, checkpoint = load_network()
    classes = checkpoint["classes"]

    # preprocess exactly as during training
    train_loader, test_loader, data_classes = get_loaders(normalize=checkpoint["normalize"])
    if data_classes != classes:
        raise ValueError(f"class mismatch: checkpoint has {classes}, data has {data_classes}")

    train_result = evaluate(net, train_loader, classes)
    test_result = evaluate(net, test_loader, classes)
    train_counts = count_images(train_loader, classes)
    test_counts = count_images(test_loader, classes)

    print(f"Train accuracy: {train_result['accuracy']:.1%}")
    print(f"Test accuracy:  {test_result['accuracy']:.1%}\n")

    print("| Class | Train images | Test images | Test accuracy |")
    print("|-------|--------------|-------------|---------------|")
    for name in classes:
        print(f"| {name} | {train_counts[name]} | {test_counts[name]} | {test_result['per_class'][name]:.1%} |")
    print(f"| **All** | {sum(train_counts.values())} | {sum(test_counts.values())} | {test_result['accuracy']:.1%} |")

    mistakes = sorted(test_result["confusions"].items(), key=lambda item: item[1], reverse=True)
    if mistakes:
        print("\nMost common test mistakes (true class -> predicted class):")
        for (true_name, predicted_name), count in mistakes[:5]:
            print(f"  {true_name} -> {predicted_name}: {count}")
    else:
        print("\nNo mistakes on the test set.")

    with open("evaluation.json", "w") as f:
        json.dump({
            "train_accuracy": train_result["accuracy"],
            "test_accuracy": test_result["accuracy"],
            "test_per_class": test_result["per_class"],
            "train_images": train_counts,
            "test_images": test_counts,
        }, f, indent=4)
    print("\nSaved evaluation.json")