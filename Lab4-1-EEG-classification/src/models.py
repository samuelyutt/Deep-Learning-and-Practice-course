import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def __str__(self):
        return f'{self.__class__.__name__}_{self.activation.__class__.__name__}'

    def load(self, path, device):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict['model_state_dict'])

    def save(self, path):
        state_dict = {'model_state_dict': self.state_dict()}
        torch.save(state_dict, path)


class EEGNet(Net):
    def __init__(self, activation, dropout=0.5):
        super(EEGNet, self).__init__()
        self.dropout = dropout

        if activation == 'ELU':
            self.activation = nn.ELU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            self.activation, 
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(dropout)
        )
        self.seperableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            self.activation, 
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(dropout)
        )
        self.classify = nn.Sequential(
            nn.Linear(736, 2, bias=True)
        )

    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseConv(out)
        out = self.seperableConv(out)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        return out


class DeepConvNet(Net):
    def __init__(self, activation, dropout=0.5):
        super(DeepConvNet, self).__init__()
        self.dropout = dropout
        if activation == 'ELU':
            self.activation = nn.ELU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        kernel_sizes = [(2, 1), (1, 5), (1, 5), (1, 5)]
        filters = [25, 25, 50, 100, 200]

        self.block0 = nn.Conv2d(1, 25, kernel_size=(1, 5))
        for i in range(1, 5):
            setattr(self, f'block{i}', nn.Sequential(
                nn.Conv2d(filters[i - 1], filters[i], kernel_sizes[i - 1]),
                nn.BatchNorm2d(filters[i], eps=1e-5, momentum=0.1),
                self.activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(self.dropout)
            ))
        self.classify = nn.Sequential(
            nn.Linear(8600, 2, bias=True)
        )

    def forward(self, x):
        out = self.block0(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        return out
