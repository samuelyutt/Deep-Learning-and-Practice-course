import numpy as np
import data
import utils


class Neuron():
    def __init__(self, weights_cnt: int, activation) -> None:
        # self.weights = [0.0 for _ in range(weights_cnt)]
        self.weights = [np.random.randn() for _ in range(weights_cnt)]
        self.activation = activation
        self.out_val = 0.0
        self.bp_tmps = [0.0 for _ in range(weights_cnt)]

    def __str__(self):
        return f'Neuron {self.weights} {self.activation.__name__} {self.out_val}\n'

    def forward(self, in_vals, bias):
        out_val = np.dot(in_vals, self.weights) + bias
        self.out_val = self.activation.apply(out_val)
        return self.out_val

    def get_out_val(self):
        return self.out_val


class Layer():
    def __init__(self, neuron_cnt: int, weight_cnt: int, activation) -> None:
        self.neurons = [Neuron(weight_cnt, activation) for _ in range(neuron_cnt)]
        self.bias = 0.0

    def __len__(self):
        return len(self.neurons)

    def __getitem__(self, key) -> Neuron:
        try:
            return self.neurons[key]
        except:
            raise IndexError

    def __str__(self):
        ret = ''
        for neuron in self:
            ret += str(neuron)
        ret += f'Bias {self.bias}\n'
        return ret

    def get_out_vals(self):
        return [neuron.get_out_val() for neuron in self]


class FCNet():
    def __init__(self, neuron_cnts: list, activation, loss) -> None:
        self.layers = []
        self.loss = loss
        self.activation = activation
        for l in range(len(neuron_cnts)):
            self.layers.append(Layer(
                neuron_cnt=neuron_cnts[l],
                weight_cnt=neuron_cnts[l - 1] if l > 0 else 0,
                activation=activation
            ))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, key) -> Layer:
        try:
            return self.layers[key]
        except:
            raise IndexError

    def __str__(self):
        ret = 'FCNet\n'
        for l, layer in enumerate(self):
            ret += f'Layer {l}\n'
            ret += str(layer)
        return ret

    def forward(self, in_vals: list) -> list:
        for i, in_val in enumerate(in_vals):
            self[0][i].out_val = in_val

        for l, layer in enumerate(self):
            if l == 0:
                continue
            for neuron in layer:
                neuron.forward(self[l - 1].get_out_vals(), layer.bias)

        return self[-1].get_out_vals()

    def cal_loss(self, out_vals: list, gt_vals: list) -> float:
        return self.loss.apply(gt_vals, out_vals)

    def backward(self, out_vals, gt_vals, lr=0.5) -> None:
        # Output layer
        for n, neuron in enumerate(self[-1]):
            tmp2 = neuron.activation.derivative(neuron.out_val)
            for w, weight in enumerate(neuron.weights):
                tmp1 = self.loss.partial_derivative(gt_vals, out_vals, n)
                tmp3 = self[-2][w].get_out_val()
                tmp4 = weight
                part_der = tmp1 * tmp2 * tmp3
                neuron.bp_tmps[w] = tmp1 * tmp2 * tmp4
                neuron.weights[w] -= lr * part_der

        # Hidden layers
        for l in range(len(self) - 2, 0, -1):
            for n, neuron in enumerate(self[l]):
                tmp2 = neuron.activation.derivative(neuron.out_val)
                for w, weight in enumerate(neuron.weights):
                    tmp1 = np.sum([next_neuron.bp_tmps[n] for next_neuron in self[l + 1]])
                    tmp3 = self[l - 1][w].get_out_val()
                    tmp4 = weight
                    part_der = tmp1 * tmp2 * tmp3
                    neuron.bp_tmps[w] = tmp1 * tmp2 * tmp4
                    neuron.weights[w] -= lr * part_der


def inference(net: FCNet, dataset: tuple, epoch: int, lr: float=0.1, **kwargs) -> None:
    x, y = dataset

    # Train
    epoch_losses = []
    for e in range(1, epoch + 1):
        losses = []
        for in_vals, gt_vals in zip(x, y):
            out_vals = net.forward(in_vals)
            loss = net.cal_loss(out_vals, gt_vals)
            net.backward(out_vals, gt_vals, lr=lr)
            losses.append(loss)
        epoch_loss = np.mean(losses)
        epoch_losses.append(epoch_loss)
        if e % kwargs.get('print_loss_per_epoch', 1) == 0:
            print(f'epoch {e} loss : {epoch_loss}')
    print(net)

    # Predict
    pred_y = []
    out_y = []
    for in_vals in x:
        out_vals = net.forward(in_vals)
        out_y.append(out_vals)
        if out_vals[0] > 0.5:
            pred_y.append(1)
        else:
            pred_y.append(0)
    pred_y = np.array(pred_y).reshape(len(pred_y), 1)
    out_y = np.array(out_y).reshape(len(out_y), 1)

    # Show results
    acc = np.mean((y == pred_y).flatten())
    print(out_y)
    print(f'Accuracy: {acc}')
    utils.show_result(
        x, y, pred_y, epoch_losses,
        Data=kwargs.get('dataset_name', '?'),
        Activation=net.activation.__name__,
        Loss=net.loss.__name__,
        lr=lr,
        ACC=acc,
    )


def main() -> None:
    linear_dataset = data.generate_linear(random_seed=2)
    XOR_easy_dataset = data.generate_XOR_easy()

    # General
    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.1,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=XOR_easy_dataset,
        epoch=3000,
        lr=0.1,
        print_loss_per_epoch=300, dataset_name='XOR_easy',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MAELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.1,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MAELoss),
        dataset=XOR_easy_dataset,
        epoch=15000,
        lr=0.05,
        print_loss_per_epoch=300, dataset_name='XOR_easy',
    )

    # Try different learning rates
    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=1.0,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.01,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.001,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.0001,
        print_loss_per_epoch=50, dataset_name='linear',
    )

    # Try different numbers of hidden units
    inference(
        net=FCNet([2, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.1,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.1,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.1,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 4, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.1,
        print_loss_per_epoch=50, dataset_name='linear',
    )

    # Try without activation functions
    inference(
        net=FCNet([2, 4, 4, 1], utils.NoActivation, utils.MSELoss),
        dataset=linear_dataset,
        epoch=100,
        lr=0.01,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.NoActivation, utils.MSELoss),
        dataset=linear_dataset,
        epoch=100,
        lr=0.001,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.NoActivation, utils.MSELoss),
        dataset=linear_dataset,
        epoch=100,
        lr=0.0001,
        print_loss_per_epoch=50, dataset_name='linear',
    )

    # Try different activation functions
    inference(
        net=FCNet([2, 4, 4, 1], utils.ReLu, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.01,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.ReLu, utils.MSELoss),
        dataset=XOR_easy_dataset,
        epoch=3000,
        lr=0.01,
        print_loss_per_epoch=300, dataset_name='XOR_easy',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.TanH, utils.MSELoss),
        dataset=linear_dataset,
        epoch=500,
        lr=0.01,
        print_loss_per_epoch=50, dataset_name='linear',
    )
    inference(
        net=FCNet([2, 4, 4, 1], utils.TanH, utils.MSELoss),
        dataset=XOR_easy_dataset,
        epoch=2000,
        lr=0.01,
        print_loss_per_epoch=300, dataset_name='XOR_easy',
    )


if __name__ == '__main__':
    main()
