import typing

import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from abc import ABC, abstractmethod
import pickle
from os.path import join

NUM_CLASSES = 10
NUM_CLASS_SAMPLES = 4
IMG_DIM = 28


#############################################
# Helper functions
#############################################
def plot_classes(img_by_class: Dict, num_samples: int = NUM_CLASS_SAMPLES) -> None:
    # Creating a grid of (NUM_CLASSES)X(SAMPLES PER CLASS)
    fig, ax = plt.subplots(nrows=NUM_CLASSES, ncols=(1 + NUM_CLASS_SAMPLES), figsize=(10, 40))

    # Assigning the images to display
    for k in sorted(img_by_class.keys()):
        for i in range(num_samples):
            ax[k, i + 1].imshow(img_by_class[k][i][1:].reshape(IMG_DIM, IMG_DIM), cmap='gray')
        ax[k, 0].text(0.5, 0.5, f'label: {k}', fontsize=20)

    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()


def visualize_data(data: pd.DataFrame, num_classes: int = NUM_CLASSES, num_samples: int = NUM_CLASS_SAMPLES) -> None:
    img_by_class = {}
    for c in range(num_classes):
        # Getting only samples with the current label
        data_class = data[data['label'] == c]

        # Selecting random samples from the class
        class_samples = np.random.randint(0, data_class.shape[0], size=NUM_CLASS_SAMPLES)
        img_by_class[c] = []
        for i in class_samples:
            img_by_class[c].append(data_class.iloc[i, :].values)

    # PLotting the samples of each class
    plot_classes(img_by_class=img_by_class)


def train_test_split(data: pd.DataFrame, ratio: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_size = data.shape[0]
    train_size = int(data_size * ratio)

    # Choosing the entries for the train split according to the requested split ratio
    train_entries = np.random.choice(data_size, size=train_size, replace=False)

    # Setting the indices for the test split
    mask = np.isin(np.arange(data_size), train_entries)
    test_entries = np.arange(data_size)[~mask]

    # Creating the two splits
    train_set = data.iloc[train_entries, :]
    test_set = data.iloc[test_entries, :]

    # Extract the samples and labels for the train and test splits
    x_train = train_set.iloc[:, 1:].T.values
    y_train = train_set.iloc[:, 0].values
    x_test = test_set.iloc[:, 1:].T.values
    y_test = test_set.iloc[:, 0].values

    return x_train, y_train, x_test, y_test


def normalize_data(data: np.array, min_val: float = None, max_val: float = None) -> Tuple[np.array, float, float]:
    if min_val is None or max_val is None:
        min_val = np.min(data, axis=1)
        max_val = np.max(data, axis=1)

    norm_data = (data.T - min_val) / (max_val - min_val)
    return norm_data.T, min_val, max_val


def encode_one_hot_labels(labels: np.array) -> np.array:
    # retrieving the output size of each one-hot vector
    unique_labels = np.unique(labels)

    # Creating the output array of the one-hot labels
    one_hot_labels = np.zeros((labels.shape[0], unique_labels.shape[0])).astype(np.int)

    # Setting the single '1' in the one-hot vector based on the label
    for idx, l in enumerate(labels):
        one_hot_labels[idx, l] = 1

    return one_hot_labels


#############################################
# Loss
#############################################
class RegCELoss:
    def __init__(self, learning_rate: float, regularization_factor: float = 0.0):
        super().__init__()

        self.weights = None
        self.learning_rate = learning_rate
        self.reg_factor = regularization_factor

    def forward(self, logist: np.array, y_gt: np.array, w: Dict) -> float:
        # Saving the input weights
        self.weights = w

        # Calculating the cross-entropy loss
        num_samples = logist.shape[1]
        loss = -1 * np.sum(np.log(logist + 1e-6) * y_gt) / num_samples

        # Adding the regularization term to the cross-entropy loss
        loss_reg_term = 0
        for k in self.weights.keys():
            if 'b' not in k:
                loss_reg_term += np.linalg.norm(self.weights[k])
        loss_reg_term *= self.reg_factor
        loss += loss_reg_term

        return loss

    def lr_backprop(self, y_est: np.array, y: np.array, x: np.array) -> float:
        num_samples = y_est.shape[1]
        dl_dw = -1 * ((y - y_est) @ x.T) / num_samples

        # Adding the regularization term to the cross-entropy loss
        dl_reg_term = 2 * self.reg_factor * self.weights['w']
        dl_dw += dl_reg_term

        self.weights['w'] -= self.learning_rate * dl_dw

        return self.weights['w']

    def nn_backprop(self, y_est: np.array, y: np.array, x: np.array, network_params: Dict) -> Tuple[
        np.array, np.array, np.array, np.array]:
        # Getting the number of samples
        num_samples = y_est.shape[1]

        # Compute gradient of the loss
        dl_dz2 = y_est - y
        dl_db2 = (1. / num_samples) * np.sum(dl_dz2, axis=1, keepdims=True)
        dl_dw2 = (1. / num_samples) * np.matmul(dl_dz2, network_params['h'].T)
        dl_dh = np.matmul(self.weights['w2'].T, dl_dz2)

        # Calculating the loss based on the selected activation function
        dl_dz1 = dl_dh * network_params['activation_func'].derivative(network_params['z1'])

        dl_dw1 = (1. / num_samples) * np.matmul(dl_dz1, x.T)
        dl_db1 = (1. / num_samples) * np.sum(dl_dz1, axis=1, keepdims=True)

        # Backpropagation over the network's weights and biases
        self.weights['w1'] -= self.learning_rate * (dl_dw1 + 2 * self.reg_factor * self.weights['w1'])
        self.weights['b1'] -= self.learning_rate * dl_db1
        self.weights['w2'] -= self.learning_rate * (dl_dw2 + 2 * self.reg_factor * self.weights['w2'])
        self.weights['b2'] -= self.learning_rate * dl_db2

        return self.weights['w1'], self.weights['b1'], self.weights['w2'], self.weights['b2']


#############################################
# Activation Functions
#############################################
class ActivationFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(x: np.array) -> np.array:
        pass

    @abstractmethod
    def derivative(x) -> np.array:
        pass


class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: np.array) -> np.array:
        # ReLU activation:
        # x := x for x > 0 or 0 for x <= 0
        return np.maximum(x, 0)

    @staticmethod
    def derivative(x) -> np.array:
        # Setting the baseline output to zero
        dev = np.zeros_like(x)

        # Setting the derivative to 1 only for x > 0
        dev[x > 0] = 1
        return dev


class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.array) -> np.array:
        sig_x = self.forward(x)
        return sig_x * (1 - sig_x)


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: np.array) -> np.array:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(self, x: np.array) -> np.array:
        tanh_x = self.forward(x)
        return 1 - (tanh_x ** 2)


def softmax(logist: np.array) -> np.array:
    c = np.max(logist, axis=0).reshape(1, logist.shape[1])
    x_mins_max = logist - np.ones((logist.shape[0], 1)) @ c
    res = np.exp(x_mins_max) / np.sum(np.exp(x_mins_max), axis=0).reshape(1, logist.shape[1])
    return res


#############################################
# Models
#############################################

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def get_weights_dict(self):
        pass

    @abstractmethod
    def update_weights(self):
        pass

    @abstractmethod
    def get_products_dict(self):
        pass


class LogisticRegressor(Model):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # Initializing the model's weights
        self.w = np.random.rand(output_dim, input_dim)

    def forward(self, x: np.array) -> np.array:
        z = softmax(self.w @ x)
        return z

    def get_weights_dict(self):
        return {'w': self.w}

    def update_weights(self, new_w):
        self.w = new_w

    def get_products_dict(self):
        return {'z': self.z}


class NeuralNetwork(Model):
    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 act_func: ActivationFunction,
                 apply_dropout: bool):
        super().__init__()

        # Initializing the model's weights
        self.w1 = np.random.rand(hidden_dim, input_dim)
        self.b1 = np.random.rand(hidden_dim, 1)
        self.w2 = np.random.rand(output_dim, hidden_dim)
        self.b2 = np.random.rand(output_dim, 1)
        self.activation = act_func
        self.apply_dropout = apply_dropout
        self.dropout_prop = 0.2

        self.z1 = None
        self.h = None
        self.z2 = None

    def dropout(self, x):
        drop_mat = np.random.binomial([np.ones_like(x)], 1 - self.dropout_prop)[0] * (1.0 / (1 - self.dropout_prop))
        x *= drop_mat
        return x

    def forward(self, x: np.array) -> np.array:
        self.z1 = self.w1 @ x + self.b1
        self.h = self.activation.forward(self.z1)

        if self.apply_dropout:
            self.h = self.dropout(self.h)

        self.z2 = self.w2 @ self.h + self.b2
        y_est = softmax(self.z2)
        return y_est

    def get_weights_dict(self):
        return {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}

    def update_weights(self, new_w1, new_b1, new_w2, new_b2):
        self.w1 = new_w1
        self.b1 = new_b1
        self.w2 = new_w2
        self.b2 = new_b2

    def get_products_dict(self):
        return {'activation_func': self.activation, 'z1': self.z1, 'h': self.h, 'z2': self.z2}


#############################################
def run_validation(x: np.array, y: np.array, lambda_reg: float, model: Model) -> Tuple[float, float]:
    # Calculating the model's outputs
    z = model.forward(x)

    # Retrieving the predicted labels
    preds = np.argmax(z, axis=0)

    # Define the loss class
    regularized_cross_entropy_loss = RegCELoss(learning_rate=None, regularization_factor=lambda_reg)

    test_loss = regularized_cross_entropy_loss.forward(z, y, model.get_weights_dict())
    accuracy = np.mean(preds == np.argmax(y, axis=0))
    return accuracy, test_loss


def logistic_regression(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, num_epochs: int,
                        learning_rate: float, batch_size: int, regularization_factor: float) -> Tuple[
    np.array, np.array, np.array, np.array]:
    # Normalizing the training data
    x_train, min_val, max_val = normalize_data(x_train)
    x_test, _, _ = normalize_data(x_test, min_val, max_val)
    x_test = np.concatenate((x_test, np.ones((1, x_test.shape[1]))), axis=0)

    # Encode the labels using one-hot
    y_train_oh = encode_one_hot_labels(y_train).T
    y_test_oh = encode_one_hot_labels(y_test).T

    # Initializing the logistic regression model
    lr_model = LogisticRegressor(x_train.shape[0] + 1, NUM_CLASSES)

    # Define the loss class
    regularized_cross_entropy_loss = RegCELoss(learning_rate=learning_rate,
                                               regularization_factor=regularization_factor)

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for batch_idx, index in enumerate(range(0, x_train.shape[0], batch_size)):
            index_end = min(index + batch_size, x_train.shape[0])
            x = x_train[:, index:index_end]
            x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)
            y = y_train_oh[:, index:index_end]

            # matrix-vector multiplication
            z = lr_model.forward(x)

            # calculate loss
            criteria = regularized_cross_entropy_loss.forward(z, y, lr_model.get_weights_dict())
            epoch_loss += criteria

            # Calculate accuracy
            y_est = np.argmax(z, axis=0)
            epoch_accuracy += np.mean(y_est == np.argmax(y, axis=0))

            # Backpropagation over the model weights
            lr_model.update_weights(regularized_cross_entropy_loss.lr_backprop(z, y, x))

        # Run model evaluation on the validation set
        val_epoch_accuracy, val_epoch_loss = run_validation(x_test, y_test_oh, regularization_factor, lr_model)
        print(f'Validation - Epoch {epoch}: Loss={val_epoch_loss}, Accuracy={val_epoch_accuracy}')

        # Saving the epoch losses and accuracies
        train_loss.append(epoch_loss / (batch_idx + 1))
        train_accuracy.append(epoch_accuracy / (batch_idx + 1))
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    # Saving the model
    filename = join('out', f'logreg_model_lr{learning_rate}_bs{batch_size}_reg{regularization_factor}.pkl')
    pickle.dump(lr_model, open(filename, 'wb'))

    return train_loss, train_accuracy, val_loss, val_accuracy


def neural_network(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, num_epochs: int,
                   hidden_dim: int, learning_rate: float, batch_size: int, regularization_factor: float,
                   activation_func: ActivationFunction, apply_dropout: bool) -> Tuple[
    np.array, np.array, np.array, np.array]:
    # Normalizing the training data
    x_train, min_val, max_val = normalize_data(x_train)
    x_test, _, _ = normalize_data(x_test, min_val, max_val)

    # Encode the labels using one-hot
    y_train_oh = encode_one_hot_labels(y_train).T
    y_test_oh = encode_one_hot_labels(y_test).T

    # Initializing the neural-network model
    nn_model = NeuralNetwork(input_dim=x_train.shape[0],
                             hidden_dim=hidden_dim,
                             output_dim=NUM_CLASSES,
                             act_func=activation_func,
                             apply_dropout=apply_dropout)

    # Define the loss class
    regularized_cross_entropy_loss = RegCELoss(learning_rate=learning_rate,
                                               regularization_factor=regularization_factor)

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for batch_idx, index in enumerate(range(0, x_train.shape[0], batch_size)):
            index_end = min(index + batch_size, x_train.shape[0])
            x = x_train[:, index:index_end]
            y = y_train_oh[:, index:index_end]

            # Calculating the neural-network's estimation
            y_est = nn_model.forward(x)

            # Calculate the regularized CE loss
            criteria = regularized_cross_entropy_loss.forward(y_est, y, nn_model.get_weights_dict())
            epoch_loss += criteria

            # Calculate accuracy
            preds = np.argmax(y_est, axis=0)
            epoch_accuracy += np.mean(preds == np.argmax(y, axis=0))

            # Backpropagation over the network's weights and biases
            nn_model.update_weights(
                *regularized_cross_entropy_loss.nn_backprop(y_est, y, x, nn_model.get_products_dict()))

        # Run model evaluation on the validation set
        val_epoch_accuracy, val_epoch_loss = run_validation(x_test, y_test_oh, regularization_factor, nn_model)
        print(f'Validation - Epoch {epoch}: Loss={val_epoch_loss}, Accuracy={val_epoch_accuracy}')

        # Saving the epoch losses and accuracies
        train_loss.append(epoch_loss / (batch_idx + 1))
        train_accuracy.append(epoch_accuracy / (batch_idx + 1))
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    # Saving the model
    filename = join('out', f'nn_model_hid{hidden_dim}_lr{learning_rate}_bs{batch_size}_reg{regularization_factor}.pkl')
    pickle.dump(nn_model, open(filename, 'wb'))

    return train_loss, train_accuracy, val_loss, val_accuracy


def plot_losses_and_accuracies(train_loss: List, train_accuracy: List, val_loss: List, val_accuracy: List) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(train_loss, label='Train Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_title('Train and Validation Losses')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(train_accuracy, label='Train Accuracy')
    ax[1].plot(val_accuracy, label='Validation Accuracy')
    ax[1].set_title('Train and Validation Model Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].grid(True)
    ax[1].legend()
    plt.show()


def plot_multiple_losses_and_accuracies(experiments: Dict, disp_factor: int = 10) -> None:
    colors = cm.rainbow(np.linspace(0, 1, len(experiments.keys())))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    for idx, k in enumerate(experiments.keys()):
        for values in experiments[k].keys():
            marker = ','
            if "validation" in values.lower():
                marker = 'v'
            x_axis = np.arange(0, len(experiments[k][values]), disp_factor)

            if "loss" in values.lower():
                ax[0].plot(x_axis, experiments[k][values][::disp_factor], label=values, color=colors[idx],
                           marker=marker)
            elif "accuracy" in values.lower():
                ax[1].plot(x_axis, experiments[k][values][::disp_factor], label=values, color=colors[idx],
                           marker=marker)
    ax[0].set_title('Train and Validation Losses')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title('Train and Validation Model Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].grid(True)
    ax[1].legend()
    plt.savefig(f'experiments.png')
    plt.show()


def main():
    # Reading the training data
    train_data = pd.read_csv(r'train.csv')

    ##########################################################
    # Part 1 - Visualize the Data
    ##########################################################
    # visualize_data(train_data)

    ##########################################################
    # Preprocessing the data
    ##########################################################
    # Splitting the training dataset to train and test split
    x_train, y_train, x_test, y_test = train_test_split(train_data)

    ##########################################################
    # Part 2 - Logistic Regression Classifier
    ##########################################################
    experiments = {}
    batch_sizes = [8, 16, 32, 64, 128, 256]
    reg_factor = [0.0, 0.5, 0.01, 0.0001]
    lr = [1.0, 0.1, 0.01, 0.001, 0.0001]
    exp_name = 'Learning Rate'
    for param in lr:
        train_loss, train_accuracy, val_loss, val_accuracy = logistic_regression(x_train=x_train,
                                                                                 y_train=y_train,
                                                                                 x_test=x_test,
                                                                                 y_test=y_test,
                                                                                 num_epochs=200,
                                                                                 learning_rate=param,
                                                                                 batch_size=16,
                                                                                 regularization_factor=0.005)
        experiments[param] = {f"Train Loss ({exp_name}={param})": train_loss,
                              f"Train Accuracy ({exp_name}={param})": train_accuracy,
                              f"Validation Loss ({exp_name}={param})": val_loss,
                              f"Validation Accuracy ({exp_name}={param})": val_accuracy}
    plot_multiple_losses_and_accuracies(experiments, disp_factor=10)

    ##########################################################
    # Part 3 - NN with a Softmax Activation
    ##########################################################
    experiments = {}
    batch_sizes = [8, 16, 32, 64, 128, 256]
    reg_factor = [0.0, 0.5, 0.01, 0.005, 0.0001]
    lr = [1.0, 0.1, 0.01, 0.001, 0.0001]
    hidden_dim = [32, 64, 128, 256, 512, 1024]
    activation_funcs = [ReLU(), Sigmoid(), Tanh()]
    apply_dropout = [False, True]

    exp_name = 'Dropout'
    for idx, param in enumerate(apply_dropout):
        train_loss, train_accuracy, val_loss, val_accuracy = neural_network(x_train=x_train,
                                                                            y_train=y_train,
                                                                            x_test=x_test,
                                                                            y_test=y_test,
                                                                            num_epochs=1000,
                                                                            hidden_dim=32,
                                                                            learning_rate=0.01,
                                                                            batch_size=64,
                                                                            regularization_factor=0.005,
                                                                            activation_func=ReLU(),
                                                                            apply_dropout=param)

        experiments[param] = {f"Train Loss ({exp_name}={param})": train_loss,
                              f"Train Accuracy ({exp_name}={param})": train_accuracy,
                              f"Validation Loss ({exp_name}={param})": val_loss,
                              f"Validation Accuracy ({exp_name}={param})": val_accuracy}
    plot_multiple_losses_and_accuracies(experiments, disp_factor=10)

    ##########################################################
    # Part 4 - Models Evaluation
    ##########################################################
    # Reading the test data
    test_data = pd.read_csv(r'test.csv')
    x_test = test_data.T.values

    # Loading the best trained logistic regression model and running inference over the test data
    best_lr_model = pickle.load(open(join('out', 'logreg_model_lr0.01_bs8_reg0.005.pkl'), 'rb'))
    x_test_lr = np.concatenate((x_test, np.ones((1, x_test.shape[1]))), axis=0)
    lr_res = best_lr_model.forward(x_test_lr)
    lr_preds = np.argmax(lr_res, axis=0)
    model_pred = pd.DataFrame(lr_preds)
    model_pred.to_csv('lr_pred.csv', header=False ,index=False)

    # Loading the best trained neural network model and running inference over the test data
    best_nn_model = pickle.load(open(join('out', 'nn_model_hid32_lr0.01_bs16_reg0.005.pkl'), 'rb'))
    nn_res = best_nn_model.forward(x_test)
    nn_preds = np.argmax(nn_res, axis=0)
    model_pred = pd.DataFrame(nn_preds)
    model_pred.to_csv('NN_pred.csv', header=False ,index=False)


if __name__ == "__main__":
    main()
