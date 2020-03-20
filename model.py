import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import pandas as pd


class VGGNetwork(nn.Module):
    def __init__(self):
        super(VGGNetwork, self).__init__()
        self.cnv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
        )
        self.fc1 = nn.Linear(512, 1024)
        self.out = nn.Linear(1024, 7)

    def forward(self, t):
        t = self.cnv_layers(t)
        t = t.view(t.size(0), -1)
        t = self.fc1(t)
        t = F.relu(t, inplace=True)
        t = F.dropout(t, p=0.5, training=self.training)

        t = self.out(t)
        return t


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def load_data():
    width, height = 48, 48
    data = pd.read_csv('fer2013.csv')  # I didn't include this on git, there is a link to dataset in README
    emotions = data['emotion'].tolist()
    pixels = data['pixels'].tolist()
    usages = data['Usage'].tolist()
    y_train = []
    y_test = []
    x_train = []
    x_test = []
    i = 0
    for ps in pixels:
        temp = [int(p) for p in ps.split(' ')]
        temp = torch.Tensor(temp).reshape((width, height)).unsqueeze(dim=0)
        if usages[i] == 'Training':
            x_train.append(temp)
            y_train.append(emotions[i])
        else:
            x_test.append(temp)
            y_test.append(emotions[i])
        i += 1
    x_train = torch.stack(x_train, dim=0) / 255
    x_test = torch.stack(x_test, dim=0) / 255
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    print('Training set size: ' + str(x_train.shape[0]))
    print('Test set size: ' + str(x_test.shape[0]))
    return x_train, x_test, y_train, y_test


def train_model():
    x_train, x_test, y_train, y_test = load_data()
    total_size = x_train.shape[0]
    dataset = data.TensorDataset(x_train, y_train)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    network = VGGNetwork()
    network.cuda()
    optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=10e-5)

    for epoch in range(40):
        total_loss = 0
        total_correct = 0
        for batch in data_loader:
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()
            preds = network(images)

            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
        print("epoch: ", epoch, "total_correct: ", total_correct, "loss:", total_loss)
        print("Accuracy: " + str(total_correct / total_size))
        print('--------------------------------------------------------------------')
    torch.save(network.state_dict(), 'model.pth')


def test_model():
    x_train, x_test, y_train, y_test = load_data()
    print(x_test.shape)
    network = VGGNetwork()
    network.load_state_dict(torch.load('model.pth'))
    dataset = data.TensorDataset(x_test, y_test)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    total_size = x_test.shape[0]
    total_loss = 0
    total_correct = 0
    for batch in data_loader:
        images, labels = batch
        preds = network(images)
        total_correct += get_num_correct(preds, labels)
    print("Accuracy: " + str(total_correct / total_size))
