import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m


class CNN(nn.Module):

    O_1 = 17
    O_2 = 18
    O_3 = 32
    O_4 = 37

    K_1 = 3
    K_2 = 1
    K_3 = 4
    K_4 = 2

    KP_1 = 4
    KP_2 = 4
    KP_3 = 1
    KP_4 = 1

    reshape = 141
    conv_linear_out = int(m.floor(
        (m.floor((m.floor((m.floor((m.floor((reshape - K_1 + 1) / KP_1) -
                                    K_2 + 1) / KP_2) - K_3 + 1) / KP_3) -
                                    K_4 + 1) / KP_4)**2) * O_4))

    FN_1 = 148

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, self.O_1, self.K_1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(self.KP_1))
        self.conv2 = nn.Sequential(nn.Conv2d(self.O_1, self.O_2, self.K_2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(self.KP_2))
        self.conv3 = nn.Sequential(nn.Conv2d(self.O_2, self.O_3, self.K_3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(self.KP_3))
        self.conv4 = nn.Sequential(nn.Conv2d(self.O_3, self.O_4, self.K_4),
                                   nn.ReLU(),
                                   nn.MaxPool2d(self.KP_4))
        self.fc1 = nn.Linear(self.conv_linear_out, self.FN_1, nn.Dropout(0.2))
        self.fc2 = nn.Linear(self.FN_1, 10)

    def forward(self, x):
        x = x.float()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(len(x), -1)
        x = F.logsigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def load_cnn(path):
    cnn = CNN()
    cnn = torch.load(path)
    return cnn
