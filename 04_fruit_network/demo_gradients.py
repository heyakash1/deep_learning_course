import math
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders, IMAGE_SIZE
import model


def show_one_update(net: nn.Module, images: torch.Tensor, labels: torch.Tensor, lr: float = 0.01) -> None:
    """Performs a single training step demonstration, illustrating gradient backpropagation, parameter updates, and loss change."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    optimizer.zero_grad()

    # Show that gradients are None before backward() fills them in
    print(f"Gradient before backward(): {net.fc4.weight.grad}")

    # Forward pass and loss computation
    outputs = net(images)
    initial_loss = criterion(outputs, labels)

    # Backpropagation
    initial_loss.backward()

    # Pick the weight with the largest absolute gradient so a dead ReLU doesn't hide the update
    grad_abs = net.fc4.weight.grad.abs()
    max_idx = divmod(torch.argmax(grad_abs).item(), grad_abs.shape[1])

    grad_val = net.fc4.weight.grad[max_idx].item()
    initial_weight = net.fc4.weight[max_idx].item()

    print("\n--- Weight Update Demonstration ---")
    print(f"Initial Loss: {initial_loss.item():.6f}")
    print(f"Selected Weight Index: {max_idx}")
    print(f"Initial Weight: {initial_weight:.3e}")
    print(f"Gradient: {grad_val:.3e}")

    # Verify that backpropagation reached early layers by inspecting gradient norms
    print("\n--- Gradient Norms Across Layers ---")
    for name, param in net.named_parameters():
        if param.grad is not None:
            print(f"Layer '{name}' Gradient Norm: {param.grad.norm().item():.3e}")

    # Snapshot parameters before the step to track how many weights actually move
    params_before = {name: param.detach().clone() for name, param in net.named_parameters()}

    optimizer.step()

    # Count how many parameter entries changed across the whole network
    changed_params_count = 0
    total_params_count = 0
    for name, param in net.named_parameters():
        diff = param - params_before[name]
        changed_params_count += (diff != 0).sum().item()
        total_params_count += param.numel()

    # Re-read updated weight value directly from the parameter tensor
    updated_weight = net.fc4.weight[max_idx].item()
    actual_change = updated_weight - initial_weight
    expected_change = -lr * grad_val

    print("\n--- Post-Step Verification ---")
    print(f"Updated Weight: {updated_weight:.3e}")
    print(f"Actual Change: {actual_change:.3e}")
    print(f"Expected Change (-lr * grad): {expected_change:.3e}")

    is_matching = math.isclose(actual_change, expected_change, abs_tol=1e-7)
    print(f"Matches Expected Update (within tolerance): {is_matching}")

    # Show post-step loss reduction on the same batch under no_grad
    with torch.no_grad():
        updated_outputs = net(images)
        updated_loss = criterion(updated_outputs, labels)
        print(f"\nLoss Before Step: {initial_loss.item():.6f}")
        print(f"Loss After Step: {updated_loss.item():.6f}")

    print(f"\nNote: The optimizer step updated {changed_params_count} out of {total_params_count} parameter entries (weights with non-zero gradients).")


if __name__ == "__main__":
    # Set seed for reproducible random initialization and data shuffling
    torch.manual_seed(42)

    # Load data (using _ to ignore unused test_loader)
    train_loader, _, classes = get_loaders()

    # Instantiate model and retrieve one batch
    net = model.FruitNetwork(IMAGE_SIZE * IMAGE_SIZE, len(classes))
    images, labels = next(iter(train_loader))

    show_one_update(net, images, labels)