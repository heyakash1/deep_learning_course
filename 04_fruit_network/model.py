import torch.nn as nn


class FruitNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        # 4 linear layers: input_size -> 10 -> 10 -> 10 -> num_classes
        self.fc1 = nn.Linear(input_size,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        self.fc4 = nn.Linear(10,num_classes)

        # ReLu activation instance
        self.relu = nn.ReLU()

    def forward(self, x):
        # flattern x from (batch, 1, 20, 20) to (batch,400)
        x = x.view(x.size(0),-1)

        # Pass through linear layers with ReLu after first three
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        # Output layer without activation (raw logits for CrossEntropyLoss)
        x = self.fc4(x)

        return x
