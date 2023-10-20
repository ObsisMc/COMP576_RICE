import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork


def generate_data(dataset="make_moons"):
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    if dataset.lower() != "make_moons":
        X, y = datasets.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0),
                                   shuffle=True, random_state=None)
    else:
        X, y = datasets.make_moons(200, noise=0.20)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class DeepNeuralNetwork:
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim: int, nn_output_dim, nn_hidden_dim, layer_num, actFun_type='tanh', reg_lambda=0.01,
                 seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        assert layer_num >= 3
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.layer_num = layer_num
        self.layer_dims = [self.nn_input_dim] + [self.nn_hidden_dim] * (layer_num - 2) + [self.nn_output_dim]
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.fcs = [LinearLayer(self.layer_dims[i], self.layer_dims[i + 1]) for i in range(self.layer_num - 1)]
        self.activations = [ActivationLayer(self.actFun_type) for _ in range(self.layer_num - 2)]
        self.activations.append(ActivationLayer("identity"))

    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        for i, (fc, act) in enumerate(zip(self.fcs, self.activations)):
            X = fc.feedforward(X)
            X = act.feedforward(X)
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X)
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE

        data_loss = - np.sum(np.log(self.probs[range(num_examples), y]+1e-5))
        print(f'Accuary: {np.sum(self.probs.argmax(axis=1) == y) / len(y)}')

        # Add regulatization term to loss (optional)
        for fc in self.fcs:
            data_loss += self.reg_lambda / 2 * (np.sum(np.square(fc.W)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        da = self.probs  # (n,2)
        da[range(num_examples), y] -= 1  # (n,2)

        for i, (fc, act) in enumerate(zip(self.fcs[::-1], self.activations[::-1])):
            dz = act.backprop(da)
            da = fc.backprop(dz)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            self.backprop(X, y)

            for fi, fc in enumerate(self.fcs):
                fc.W += -epsilon * (fc.dW + self.reg_lambda * fc.W)
                fc.bias += -epsilon * fc.db

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # np.random.seed(seed)
        self.W = np.random.randn(self.input_dim, self.output_dim) / np.sqrt(self.input_dim)
        self.bias = np.zeros((1, self.output_dim))
        self.dW = self.db = None

        self.X = None

    def feedforward(self, X):
        self.X = np.copy(X)
        return X @ self.W + self.bias

    def backprop(self, grad):
        self.dW = self.X.T @ grad
        self.db = grad.sum(axis=0)
        dX = grad @ self.W.T
        return dX


class ActivationLayer:
    def __init__(self, tp: str):
        assert tp in ["relu", "sigmoid", "tanh", "identity"]
        self.tp = tp
        self.z = None

    def feedforward(self, z):
        self.z = np.copy(z)
        if self.tp.lower() == "relu":
            z[z < 0] = 0
            return z
        elif self.tp.lower() == "sigmoid":
            return 1 / (1 + np.e ** (-z))
        elif self.tp.lower() == "tanh":
            return (np.e ** z - np.e ** (-z)) / (np.e ** z + np.e ** (-z))
        elif self.tp.lower() == "identity":
            return z

    def backprop(self, grad):
        if self.tp.lower() == "relu":
            d = np.ones_like(self.z)
            d[self.z < 0] = 0
        elif self.tp.lower() == "sigmoid":
            d = np.copy(self.z)
            d = np.e ** (-d) / (1 + np.e ** (-d)) ** 2
        elif self.tp.lower() == "tanh":
            d = np.copy(self.z)
            d = 1 - ((np.e ** d - np.e ** (-d)) / (np.e ** d + np.e ** (-d))) ** 2
        else:
            d = np.ones_like(self.z)
        return grad * d


def main(dataset="make_moons"):
    # # generate and visualize Make-Moons dataset
    X, y = generate_data(dataset)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    input_dim, output_dim = X.shape[1] if len(X.shape) > 1 else 2, y.shape[1] if len(y.shape) > 1 else 2
    model = DeepNeuralNetwork(nn_input_dim=input_dim,
                              nn_output_dim=output_dim,
                              nn_hidden_dim=3,
                              layer_num=10,
                              actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()
    # main("make_blobs")
