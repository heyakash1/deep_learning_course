from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

IMAGE_SIZE = 20
BATCH_SIZE = 32


def get_loaders(data_dir: str = "data", batch_size: int = BATCH_SIZE) -> tuple:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        # convert to a tensor
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return (train_loader,test_loader,train_data.classes)

if __name__ == "__main__":
    train_loader, test_loader, classes = get_loaders()
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape, classes)