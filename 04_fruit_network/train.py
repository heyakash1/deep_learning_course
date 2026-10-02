from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 20

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("data/train", transform=transform)
test_data = datasets.ImageFolder("data/test", transform=transform)

train_loader = DataLoader(train_data,batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape, train_data.classes)