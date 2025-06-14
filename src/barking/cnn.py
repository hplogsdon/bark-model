from torch import nn


class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 Convolutional blocks, Blocks->(OUTPUT)-> Flatten Results -> Apply Linear Layer -> Softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 1 channel for grayscale
                out_channels=16,  # 16 filters
                kernel_size=3,  # average value for convolutional layers
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 16 channels for grayscale from output of conv1
                out_channels=32,  # 32 filters (doubling out from conv1)
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Flatten the results
        self.flatten = nn.Flatten()
        # Apply Linear layer
        self.linear = nn.Linear(in_features=(128 * 5 * 4), out_features=10)
        # Apply SoftMax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # Pass Data between the layers:
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Pass results to flatten
        flat = self.flatten(x)
        logits = self.linear(flat)
        predictions = self.softmax(logits)
        return predictions
