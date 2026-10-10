import torch
import torch.nn as nn
import torch.optim as optim

from data import get_loaders, IMAGE_SIZE
import model

# Settings for the final run. Chosen from the training-loss curves in
# experiments.py, not from test accuracy.
SEED = 20
NORMALIZE = True
EPOCHS = 20
LR = 0.05
CHECKPOINT = "fruit_net.pt"


def train(net, train_loader, epochs: int = EPOCHS, lr: float = LR) -> list:
    """
    Trains net and returns a list with the average loss of each epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # 1. clear the gradients left over from the previous batch
            optimizer.zero_grad()

            # 2. forward pass
            outputs = net(images)

            # 3. measure how wrong the outputs are
            loss = criterion(outputs, labels)

            # 4. backward pass (backpropagation)
            loss.backward()

            # 5. update the weights
            optimizer.step()

            # .item() turns a one-value tensor into a plain float
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

    return epoch_losses


def save_checkpoint(net, classes: list, normalize: bool, path: str = CHECKPOINT) -> None:
    """
    Saves the weights together with the settings evaluation needs, so the
    test images are preprocessed exactly like the training images were.
    """
    torch.save({
        "state_dict": net.state_dict(),
        "classes": classes,
        "normalize": normalize,
        "image_size": IMAGE_SIZE,
    }, path)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    train_loader, _, classes = get_loaders(normalize=NORMALIZE)
    net = model.FruitNetwork(IMAGE_SIZE * IMAGE_SIZE, len(classes))

    losses = train(net, train_loader)

    save_checkpoint(net, classes, NORMALIZE)
    print(f"Saved weights and settings to {CHECKPOINT}")