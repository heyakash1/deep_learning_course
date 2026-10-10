import torch
import torch.nn as nn
import torch.optim as optim

from data import IMAGE_SIZE
from evaluate import evaluate
from model import FruitNetwork
from train import train

NUM_CLASSES = 10
BATCH_SIZE = 32
CLASSES = [f"class_{i}" for i in range(NUM_CLASSES)]


def make_net(seed: int = 0) -> FruitNetwork:
    torch.manual_seed(seed)
    return FruitNetwork(IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES)


def make_batch(seed: int = 0) -> tuple:
    """A fake batch of random images and random labels, so no dataset is needed."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), generator=generator)
    return images, labels


def test_output_shape():
    net = make_net()
    images, _ = make_batch()

    outputs = net(images)

    assert outputs.shape == (BATCH_SIZE, NUM_CLASSES), f"expected {(BATCH_SIZE, NUM_CLASSES)}, got {tuple(outputs.shape)}"


def test_one_step_changes_weights():
    net = make_net()
    images, labels = make_batch()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05)
    before = net.fc1.weight.detach().clone()

    optimizer.zero_grad()
    criterion(net(images), labels).backward()
    assert net.fc1.weight.grad is not None, "no gradient reached the first layer"
    optimizer.step()

    assert not torch.equal(before, net.fc1.weight), "fc1.weight did not change after one step"


def test_loss_decreases():
    net = make_net()
    batch = make_batch()

    # a plain list of one batch works as a loader: train only loops over it and calls len()
    losses = train(net, [batch], epochs=20, lr=0.05)

    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_evaluate_counts_correct_predictions():
    net = make_net()
    images, _ = make_batch()
    with torch.no_grad():
        predicted = net(images).argmax(dim=1)

    right = evaluate(net, [(images, predicted)], CLASSES)
    assert right["accuracy"] == 1.0, f"expected 1.0, got {right['accuracy']}"

    wrong_labels = (predicted + 1) % NUM_CLASSES
    wrong = evaluate(net, [(images, wrong_labels)], CLASSES)
    assert wrong["accuracy"] == 0.0, f"expected 0.0, got {wrong['accuracy']}"
    assert sum(wrong["confusions"].values()) == BATCH_SIZE, "every wrong prediction should be recorded as a confusion"


if __name__ == "__main__":
    test_output_shape()
    test_one_step_changes_weights()
    test_loss_decreases()
    test_evaluate_counts_correct_predictions()
    print("All tests passed.")