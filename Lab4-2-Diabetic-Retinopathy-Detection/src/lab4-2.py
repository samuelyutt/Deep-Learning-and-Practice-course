import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from dataloader import RetinopathyLoader
from models import ResNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = '../data'
out_dir = '../out'
checkpoint_dir = '../checkpoint'


def add_comparision(acc, model, label):
    plt.title(f'Result Comparision({model.__class__.__name__}{model.layers})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.plot(
        [i for i in range(len(acc))],
        np.array(acc) * 100,
        label=label,
        linewidth=0.5
    )
    plt.legend(loc='upper left', fontsize=8)


def plot_comparision(title):
    plt.savefig(f'{out_dir}/{title}_comparision_figure.png')
    plt.close()


def plot_confusion(y_true, y_pred, model):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[0, 1, 2, 3, 4],
        cmap=plt.cm.Blues,
        normalize='true',
    )
    disp.ax_.set_title(f'{model.name}')

    # print(f'{model.name}')
    # print(disp.confusion_matrix)
    plt.savefig(f'{out_dir}/{model.name}_confusion_matrix.png')
    plt.close()


def train(model, train_loader, test_loader, lr=1e-3, epochs=5, criterion=nn.CrossEntropyLoss()):
    print(f'Training {model.name} on device {device}')

    # Initialize training
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    all_train_acc, all_test_acc, best_test_acc = [], [], -1.0
    best_test_acc = -1.0
    model.to(device)

    # Training
    for e in range(epochs):
        model.train()
        train_acc, train_loss = 0.0, 0.0
        for training_data in train_loader:
            data = training_data[0].to(device, dtype=torch.float)
            label = training_data[1].to(device, dtype=torch.long)
            optimizer.zero_grad()
            out = model(data)
            batch_loss = criterion(out, label)
            batch_loss.backward()
            optimizer.step()
            train_acc += out.max(dim=1)[1].eq(label).sum().item()
            train_loss += batch_loss.item()

        # Statistics
        train_acc /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        all_train_acc.append(train_acc)

        # Evaluate and logs
        test_acc, test_loss, _, _ = evaluate(model, test_loader)
        all_test_acc.append(test_acc)
        if e % 1 == 0:
            print(f'Epoch {e + 1} Train Acc: {train_acc} Train Loss: {train_loss} Test Acc: {test_acc} Test Loss: {test_loss}')

        # Save checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save(f'{checkpoint_dir}/{model.name}_BEST.pt')

    # Plots and results
    add_comparision(all_train_acc, model, f'Train({"with" if model.pretrained else "w/o"} pretraining)')
    add_comparision(all_test_acc, model, f'Test({"with" if model.pretrained else "w/o"} pretraining)')
    print(f'Best Test Acc of {model.name}: {best_test_acc}')


def evaluate(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.to(device)
    test_acc, test_loss = 0.0, 0.0
    y_trues, y_preds = [], []
    with torch.no_grad():
        model.eval()
        for testing_data in test_loader:
            data = testing_data[0].to(device, dtype=torch.float)
            label = testing_data[1].to(device, dtype=torch.long)
            out = model(data)
            batch_loss = criterion(out, label)
            y_pred = out.max(dim=1)[1]
            test_acc += y_pred.eq(label).sum().item()
            test_loss += batch_loss.item()
            y_trues += label.tolist()
            y_preds += y_pred.tolist()
        test_acc /= len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
    return test_acc, test_loss, y_trues, y_preds


def main():
    # Load data
    train_loader = DataLoader(
        RetinopathyLoader(data_dir, 'train'),
        batch_size=8, num_workers=4,
    )
    test_loader = DataLoader(
        RetinopathyLoader(data_dir, 'test'),
        batch_size=8, num_workers=4,
    )

    # ResNet18
    resnet18_pretrained = ResNet(18, pretrained=True)
    resnet18 = ResNet(18, pretrained=False)
    train(resnet18_pretrained, train_loader, test_loader, epochs=10)
    train(resnet18, train_loader, test_loader, epochs=10)
    plot_comparision('ResNet18')
    del resnet18_pretrained
    del resnet18

    # ResNet50
    resnet50_pretrained = ResNet(50, pretrained=True)
    resnet50 = ResNet(50, pretrained=False)
    train(resnet50_pretrained, train_loader, test_loader, epochs=5)
    train(resnet50, train_loader, test_loader, epochs=5)
    plot_comparision('ResNet50')
    del resnet50_pretrained
    del resnet50


def inference():
    print('Inference')
    test_loader = DataLoader(
        RetinopathyLoader(data_dir, 'test'),
        batch_size=8, num_workers=4,
    )

    # ResNet18
    model = ResNet(18)
    model.load('../checkpoint/BEST/ResNet18_Pretrained_BEST.pt', device)
    test_acc, _, y_true, y_pred = evaluate(model, test_loader)
    # plot_confusion(y_true, y_pred, model)
    print(f'{model.name} Test Acc:\t\t{test_acc}')

    # ResNet50
    model = ResNet(50)
    model.load('../checkpoint/BEST/ResNet50_Pretrained_BEST.pt', device)
    test_acc, _, y_true, y_pred = evaluate(model, test_loader)
    # plot_confusion(y_true, y_pred, model)
    print(f'{model.name} Test Acc:\t\t{test_acc}')


if __name__ == '__main__':
    # main()
    inference()
