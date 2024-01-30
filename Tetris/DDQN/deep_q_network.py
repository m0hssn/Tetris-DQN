import torch
import torch.nn as nn
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 5)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.embedding = nn.Linear(6, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        board, ind_b = state
        x = self.conv1(board)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x).flatten(1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.sigmoid(self.embedding(ind_b)) * x

        actions = self.fc3(x)

        return actions