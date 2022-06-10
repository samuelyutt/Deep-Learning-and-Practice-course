import torch
import torch.nn as nn


class Generator(torch.nn.Module):
    def __init__(self, z_size, c_size, ngf=64, nc=3):
        super(Generator, self).__init__()

        self.z_size = z_size
        self.c_size = c_size

        self.in_c = torch.nn.Sequential()
        c_linear = nn.Linear(24, c_size)
        self.in_c.add_module('c_linear', c_linear)
        torch.nn.init.normal_(c_linear.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(c_linear.bias, 0.0)
        # self.in_c.add_module('c_bn', torch.nn.BatchNorm1d(c_dim))
        self.in_c.add_module('c_actv', torch.nn.ReLU())

        self.hidden1 = self.add_hidden(1, z_size + c_size, ngf * 8, 4, 1, 0)
        self.hidden2 = self.add_hidden(2, ngf * 8, ngf * 4, 4, 2, 1)
        self.hidden3 = self.add_hidden(3, ngf * 4, ngf * 2, 4, 2, 1)
        self.hidden4 = self.add_hidden(4, ngf * 2, ngf, 4, 2, 1)

        self.out = torch.nn.Sequential()
        out = nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False)
        self.out.add_module('o_conv', out)
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        # torch.nn.init.constant_(out.bias, 0.0)
        self.out.add_module('o_actv', torch.nn.Tanh())

    def add_hidden(self, name, in_channels, out_channels, kernel_size, stride, padding):
        hidden = torch.nn.Sequential()
        h_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        hidden.add_module(f'h_conv_{name}', h_conv)
        torch.nn.init.normal_(h_conv.weight, mean=0.0, std=0.02)
        # torch.nn.init.constant_(h_conv.bias, 0.0)
        bn2d = torch.nn.BatchNorm2d(out_channels)
        hidden.add_module(f'h_bn_{name}', bn2d)
        torch.nn.init.normal_(bn2d.weight, mean=1.0, std=0.02)
        torch.nn.init.constant_(bn2d.bias, 0.0)
        hidden.add_module(f'h_actv_{name}', torch.nn.ReLU())
        return hidden

    def forward(self, z, c):
        z = z.view(-1, self.z_size, 1, 1)
        h_c = self.in_c(c).view(-1, self.c_size, 1, 1)
        x = torch.cat([z, h_c], dim=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        out = self.out(x)
        return out

    def load(self, path, device):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict['model_state_dict'])

    def save(self, path):
        state_dict = {'model_state_dict': self.state_dict()}
        torch.save(state_dict, path)


class Discriminator(torch.nn.Module):
    def __init__(self, img_shape, ndf=64, nc=4):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape

        self.in_c = torch.nn.Sequential()
        c_linear = nn.Linear(24, img_shape[0] * img_shape[1])
        self.in_c.add_module('c_linear', c_linear)
        torch.nn.init.normal_(c_linear.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(c_linear.bias, 0.0)
        self.in_c.add_module('c_actv', torch.nn.LeakyReLU(0.2))

        self.hidden1 = self.add_hidden(1, nc, ndf, 4, 2, 1, bn=False)
        self.hidden2 = self.add_hidden(2, ndf, ndf * 2, 4, 2, 1)
        self.hidden3 = self.add_hidden(3, ndf * 2, ndf * 4, 4, 2, 1)
        self.hidden4 = self.add_hidden(4, ndf * 4, ndf * 8, 4, 2, 1)

        self.out = torch.nn.Sequential()
        out = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.out.add_module('o_conv', out)
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        # torch.nn.init.constant_(out.bias, 0.0)
        self.out.add_module('o_actv', torch.nn.Sigmoid())

    def add_hidden(self, name, in_channels, out_channels, kernel_size, stride, padding, bn=True):
        hidden = torch.nn.Sequential()
        h_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        hidden.add_module(f'h_conv_{name}', h_conv)
        torch.nn.init.normal_(h_conv.weight, mean=0.0, std=0.02)
        # torch.nn.init.constant_(h_conv.bias, 0.0)
        if bn:
            bn2d = torch.nn.BatchNorm2d(out_channels)
            hidden.add_module(f'h_bn_{name}', bn2d)
            torch.nn.init.normal_(bn2d.weight, mean=1.0, std=0.02)
            torch.nn.init.constant_(bn2d.bias, 0.0)
        hidden.add_module(f'h_actv_{name}', torch.nn.LeakyReLU(0.2))
        return hidden

    def forward(self, img, c):
        h_c = self.in_c(c).view(-1, 1, self.img_shape[0], self.img_shape[1])
        x = torch.cat([img, h_c], dim=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        out = self.out(x)
        return out.view(-1)

    def load(self, path, device):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict['model_state_dict'])

    def save(self, path):
        state_dict = {'model_state_dict': self.state_dict()}
        torch.save(state_dict, path)
