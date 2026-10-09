import torch.nn as nn
import torch.optim as optim
import torch
from data import get_loaders, IMAGE_SIZE
import model


def train(net, train_loader, epochs: int = 20, lr: float = 0.02) -> list:
    """
    Trains model and returns a list with the average loss of each epoch.
    """
    criterion = nn.CrossEntropyLoss()       # the loss function (cross-entropy)
    optimizer = optim.SGD(net.parameters(), lr = lr)      # plain gradient descent over net.parameters()
    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # 1. zero the gradients
            optimizer.zero_grad()

            # 2. forward pass
            outputs = net(images)

            # 3. compute the loss
            loss = criterion(outputs,labels)

            # 4. backward pass
            loss.backward()

            # 5. optimizer step
            optimizer.step()

            # add this batch's loss to running_loss (.item() turns a scalar tensor into a float)
            running_loss += loss.item()

        # average loss for the epoch: append to epoch_loss and print it
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    return epoch_losses

if __name__ == "__main__":
    torch.manual_seed(20)
    train_loader, test_loader, classes = get_loaders()
    m = model.FruitNetwork(IMAGE_SIZE*IMAGE_SIZE, len(classes))
    losses = train(m,train_loader)
    print(losses)