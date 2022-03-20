import numpy as np
import matplotlib.pyplot as plt
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
        return f'Neuron {self.weights} {self.activation} {self.out_val}\n'

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
        self.input_size = neuron_cnts[0]
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


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth')
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result')
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()


def inference(net, dataset, epoch, lr=0.1, **kwargs) -> None:
    x, y = dataset

    # Train
    for e in range(1, epoch + 1):
        for in_vals, gt_vals in zip(x, y):
            out_vals = net.forward(in_vals)
            loss = net.cal_loss(out_vals, gt_vals)
            net.backward(out_vals, gt_vals, lr=lr)
        if e % kwargs.get('print_loss_per_epoch', 1) == 0:
            print(f'epoch {e} loss : {loss}')
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
    print(out_y)
    show_result(x, y, pred_y)


def main() -> None:
    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=data.generate_linear(),
        epoch=500,
        lr=0.1,
        print_loss_per_epoch=50,
    )

    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MSELoss),
        dataset=data.generate_XOR_easy(),
        epoch=3000,
        lr=0.1,
        print_loss_per_epoch=300,
    )

    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MAELoss),
        dataset=data.generate_linear(),
        epoch=500,
        lr=0.1,
        print_loss_per_epoch=50,
    )

    inference(
        net=FCNet([2, 4, 4, 1], utils.Sigmoid, utils.MAELoss),
        dataset=data.generate_XOR_easy(),
        epoch=15000,
        lr=0.05,
        print_loss_per_epoch=300,
    )


if __name__ == '__main__':
    main()
