import dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from models import EEGNet, DeepConvNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = '../data'
out_dir = '../out'
checkpoint_dir = '../checkpoint'
dropout = 0.25


def get_dataloader():
    train_data, train_label, test_data, test_label = dataloader.read_bci_data(data_dir)
    training_set = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
    testing_set = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
    train_loader = DataLoader(training_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(testing_set, batch_size=64, shuffle=False)
    return train_loader, test_loader


def plot(acc, model, posfix):
    plt.title(f'Activation function comparision({model.__class__.__name__})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.plot(
        [i for i in range(len(acc))],
        np.array(acc) * 100,
        label=f'{model}_{posfix}',
        linewidth=0.5
    )
    plt.legend(loc='lower right', fontsize=8)


def train(model, train_loader, test_loader, lr=1e-2, epochs=300, criterion=nn.CrossEntropyLoss()):
    print(f'Training {model} on device {device}')

    # Initialize training
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    all_train_acc, all_test_acc, best_test_acc = [], [], -1.0
    best_test_acc = -1.0
    model.to(device=device)

    # Training
    for e in range(epochs):
        model.train()
        train_acc, train_loss = 0.0, 0.0
        for training_data in train_loader:
            data = training_data[0].to(device, dtype=torch.float)
            label = training_data[1].to(device, dtype=torch.long)
            optimizer.zero_grad()
            prediction = model(data)
            batch_loss = criterion(prediction, label)
            batch_loss.backward()
            optimizer.step()
            train_acc += prediction.max(dim=1)[1].eq(label).sum().item()
            train_loss += batch_loss.item()

        # Statistics
        train_acc /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        all_train_acc.append(train_acc)

        # Evaluate and logs
        test_acc, test_loss = evaluate(model, test_loader)
        all_test_acc.append(test_acc)
        if e % 5 == 4:
            print(f'Epoch {e + 1} Train Acc: {train_acc} Train Loss: {train_loss} Test Acc: {test_acc} Test Loss: {test_loss}')

        # Save checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save(f'{checkpoint_dir}/{model}_BEST.pt')

    # Plots and results
    plot(all_train_acc, model, 'train')
    plot(all_test_acc, model, 'test')
    print(f'Best Test Acc of {model}: {best_test_acc}')

    return model


def evaluate(model, test_loader, criterion=nn.CrossEntropyLoss()):
    test_acc, test_loss = 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for testing_data in test_loader:
            data = testing_data[0].to(device, dtype=torch.float)
            label = testing_data[1].to(device, dtype=torch.long)
            prediction = model(data)
            batch_loss = criterion(prediction, label)
            test_acc += prediction.max(dim=1)[1].eq(label).sum().item()
            test_loss += batch_loss.item()
        test_acc /= len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
    return test_acc, test_loss


def main():
    train_loader, test_loader = get_dataloader()
    nets = [EEGNet, DeepConvNet]
    activations = ['ELU', 'LeakyReLU', 'ReLU']

    for net in nets:
        for activation in activations:
            model = net(activation, dropout=dropout)
            train(model, train_loader, test_loader, epochs=500, lr=10e-4)
        plt.savefig(f'{out_dir}/{net.__name__}.png')
        plt.close()


def inference():
    _, test_loader = get_dataloader()
    nets = [EEGNet, DeepConvNet]
    activations = ['ELU', 'LeakyReLU', 'ReLU']

    print('Inference')
    for net in nets:
        for activation in activations:
            model = net(activation, dropout=dropout).to(device)
            model.load(f'{checkpoint_dir}/{model}_BEST.pt', device)
            test_acc, _ = evaluate(model, test_loader)
            print(f'{model} Test Acc:\t\t{test_acc}')


if __name__ == '__main__':
    main()
    inference()
