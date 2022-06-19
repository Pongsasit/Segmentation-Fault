import torch
from torch import nn

class NormalCNN1(nn.Module):
    """
    The expected input is the sequence of average pixel with 15 bands (we can change this number)
        - input size: (batch, band, sequence) -> Ex. (1, 15, 71)
    """
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.cnn1 = torch.nn.Conv1d(15, 30, kernel_size=5, stride=2)
        self.cnn2 = torch.nn.Conv1d(30, 60, kernel_size=5, stride=2)
        self.cnn3 = torch.nn.Conv1d(60, 10, kernel_size=5, stride=2)
        self.linear = torch.nn.Linear(60, 4)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        x = self.activation(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        x = self.linear(x)
        return x

# class NormalCNN3(nn.Module):
#     """
#     The expected input is the sequence of image with 15 bands (we can change this number)
#         - input size: (batch, band, sequence, img_w, img_h) -> Ex. (1, 15, 71, 320, 320)
#     """
#     def __init__(self):
#         super().__init__()
#         self.activation = torch.nn.LeakyReLU()
#         self.cnn1 = torch.nn.Conv3d(15, 30, kernel_size=5, stride=2)
#         self.cnn2 = torch.nn.Conv3d(30, 60, kernel_size=5, stride=2)
#         self.cnn3 = torch.nn.Conv3d(60, 10, kernel_size=5, stride=2)
#         self.linear1 = torch.nn.Linear(82140, 2000)
#         self.linear2 = torch.nn.Linear(2000, 4)

#     def forward(self, x):
#         x = self.cnn1(x)
#         x = self.activation(x)
#         x = self.cnn2(x)
#         x = self.activation(x)
#         x = self.cnn3(x)
#         x = self.activation(x)
#         x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
#         x = self.linear1(x)
#         x = self.activation(x)
#         x = self.linear2(x)
#         return x

class NormalCNN3(nn.Module):
    """
    The expected input is the sequence of image with 15 bands (we can change this number)
        - input size: (batch, band, sequence, img_w, img_h) -> Ex. (1, 5, 71, 64, 64)
    """
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.cnn1 = torch.nn.Conv3d(5, 10, kernel_size=5, stride=2)
        self.cnn2 = torch.nn.Conv3d(10, 20, kernel_size=5, stride=2)
        self.cnn3 = torch.nn.Conv3d(20, 10, kernel_size=5, stride=2)
        self.linear1 = torch.nn.Linear(1250, 500)
        self.linear2 = torch.nn.Linear(500, 4)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        x = self.activation(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class NormalLSTM(nn.Module):
    """
    The expected input is the sequence of average pixel with 15 bands (we can change this number)
        - input size: (batch, sequence. band) -> Ex. (1, 71, 15)
    """
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.lstm = torch.nn.LSTM(input_size=15, hidden_size=64, num_layers=32, batch_first=True)
        self.linear1 = torch.nn.Linear(71*64, 2000)
        self.linear2 = torch.nn.Linear(2000, 4)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        x = self.linear(x)
        return x
