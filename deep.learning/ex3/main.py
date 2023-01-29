import os
import typing

import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.cm as cm
from os.path import join

import torch as torch
import torch.nn.functional as F
import torchvision as torchvision
from torchvision import transforms
from torchsummary import summary

NUM_CLASSES = 10
NUM_CLASS_SAMPLES = 4
IMG_DIM = 96


#############################################
# Helper functions
#############################################
def get_stl_datasets(path: str) -> Tuple[torchvision.datasets.STL10, torchvision.datasets.STL10]:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Reading the Training data
    train_transform = transforms.Compose([transforms.RandomCrop(64),
                                          # transforms.RandomRotation(degrees=(0, 20),
                                          #                           interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                                          transforms.ColorJitter(brightness=.1, saturation=.1, contrast=.1, hue=.1),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std),
                                          ])

    trainset = torchvision.datasets.STL10(root=path, split='train', download=True, transform=train_transform)

    # Reading the Test data
    test_transform = transforms.Compose([transforms.CenterCrop(64),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std),
                                         ])

    testset = torchvision.datasets.STL10(root=path, split='test', download=True, transform=test_transform)

    return trainset, testset


def plot_classes(img_by_class: Dict, num_classes: int = NUM_CLASSES, num_samples: int = NUM_CLASS_SAMPLES) -> None:
    # Creating a grid of (NUM_CLASSES)X(SAMPLES PER CLASS)
    fig, ax = plt.subplots(nrows=num_classes, ncols=(1 + num_samples))

    # Assigning the images to display
    for idx, k in enumerate(sorted(img_by_class.keys())):
        for i in range(num_samples):
            ax[idx, i + 1].imshow(np.moveaxis(img_by_class[k][i], [0, 1, 2], [2, 0, 1]))
        ax[idx, 0].text(0.3, 0.3, f'label: {k}', fontsize=20)

    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()


def visualize_data(dataset: torchvision.datasets, num_samples: int = NUM_CLASS_SAMPLES) -> None:
    img_by_class = {}
    for idx, c in enumerate(dataset.classes):
        # Getting only samples with the current label
        data_class = dataset.data[dataset.labels == idx]

        # Selecting random samples from the class
        class_samples = np.random.randint(0, data_class.shape[0], size=num_samples)
        img_by_class[c] = []
        for i in class_samples:
            img_by_class[c].append(data_class[i, :])

    # Plotting the samples of each class
    plot_classes(img_by_class=img_by_class)


#############################################
# Models
#############################################


class LogisticRegression(torch.nn.Module):
    """
    Model #1: Logistic regression over flattened version of the images
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class FullyConnectedNN(torch.nn.Module):
    """
    Model #2: Fully-connected NN with hidden layers over flattened version of the images
    """

    def __init__(self, input_dim: int, hidden_dims: List, dropout_rate: float, output_dim: int):
        super(FullyConnectedNN, self).__init__()

        self.layers = torch.nn.ModuleList()

        if not len(hidden_dims):
            self.layers.append(torch.nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
            self.layers.append(torch.nn.BatchNorm1d(hidden_dims[0]))
            self.layers.append(torch.nn.Dropout(p=dropout_rate))

            for idx in range(len(hidden_dims) - 1):
                self.layers.append(torch.nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
                self.layers.append(torch.nn.BatchNorm1d(hidden_dims[idx + 1]))
                self.layers.append(torch.nn.Dropout(p=dropout_rate))

            self.layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)

        for layer in self.layers:
            x = layer(x)
        return x


class CNN(torch.nn.Module):
    """
    Models #3-5: Convolutional Neural Networks
        #3: CNN with at least two convolution layers and two pooling layers followed by two fully connected layers
            and a classification layer
        #4-5: A fixed pre-trained MobileNetV2 as feature extractor followed by two fully connected layers and an
              additional classification layer
    """

    def __init__(self, backbone: str, dropout_rate: float, output_dim: int, freeze_backbone: bool = False):
        super(CNN, self).__init__()

        # Setting the selected backbone
        self.backbone = self._set_backbone(backbone)

        # Freezing the training of the backbone is indicated
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Setting the regression head
        self.fc1 = torch.nn.Linear(1000, 512)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.fc2 = torch.nn.Linear(512, 256)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        self.fc3 = torch.nn.Linear(256, output_dim)

    @staticmethod
    def _set_backbone(arch: str):
        if arch == 'MobileNetV2':
            return torchvision.models.mobilenet_v2(pretrained=True)
        if arch == 'Custom':
            backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, 256, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2),
                torch.nn.Conv2d(256, 512, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2),
                torch.nn.Conv2d(512, 512, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2),
                torch.nn.Conv2d(512, 1000, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(1000),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2)

            )

            return backbone
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def get_model(model_type: str, input_size: int, output_dim: int, config: Dict = None):
    if model_type == 'LogisticRegression':
        model = LogisticRegression(input_dim=input_size, output_dim=output_dim)
    elif model_type == 'FullyConnectedNN':
        model = FullyConnectedNN(input_dim=input_size,
                                 hidden_dims=config['Hidden Dims'],
                                 dropout_rate=config['Dropout Rate'],
                                 output_dim=output_dim)
    elif model_type == 'CNN':
        model = CNN(backbone=config['Backbone'],
                    dropout_rate=config['Dropout Rate'],
                    output_dim=output_dim,
                    freeze_backbone=config['Freeze Backbone'])
    else:
        raise ValueError
    return model


#############################################
# Visualization
#############################################


def plot_losses_and_accuracies(train_loss: List, train_accuracy: List, val_loss: List, val_accuracy: List,
                               model_type_str: str, params_str: str) -> None:
    epochs = np.arange(len(train_loss))
    fig, ax = plt.subplots(nrows=1, ncols=2)

    # Adding the loss values to the plot
    ax[0].plot(epochs, train_loss, label='Train Loss')
    ax[0].plot(epochs, val_loss, label='Validation Loss')
    ax[0].set_title('Loss vs. Epoch')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)
    ax[0].legend()

    # Adding the accuracy values to the plot
    ax_acc = ax[1].twinx()
    ax[1].get_yaxis().set_visible(False)
    ax_acc.plot(epochs, train_accuracy, label='Train Accuracy', marker='v')
    ax_acc.plot(epochs, val_accuracy, label='Validation Accuracy', marker='v')
    ax_acc.set_title('Accuracy vs. Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Loss')
    ax[1].grid(True)
    ax_acc.grid(True)
    ax_acc.legend()

    fig.suptitle('{} Model Evaluation - Loss and Accuracy\nParameters: {}'.format(model_type_str, params_str))

    # Saving the plot in full-screen size (better visualization)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    figure = plt.gcf()
    plt.draw()
    plt.pause(1)
    figure.savefig('experiments_{}_{}.png'.format(model_type_str, params_str.lower().replace(',', '_')))


def plot_multiple_losses_and_accuracies(experiments: Dict, exp_str: str = None, disp_factor: int = 10) -> None:
    colors = cm.rainbow(np.linspace(0, 1, len(experiments.keys())))

    for idx, fig_type in enumerate(['loss', 'accuracy']):
        plt.figure(idx + 1)

        for idx, k in enumerate(experiments.keys()):
            for values in experiments[k].keys():
                marker = ','
                if "validation" in values.lower():
                    marker = 'v'
                x_axis = np.arange(0, len(experiments[k][values]), disp_factor)

                if fig_type in values.lower():
                    plt.plot(x_axis, experiments[k][values][::disp_factor], label=values, color=colors[idx],
                             marker=marker)

        if exp_str is not None:
            plt.title(f'Train and Validation - {fig_type.capitalize()}')
            plt.xlabel('Epochs')
            plt.ylabel(fig_type.capitalize())
            plt.legend()
            plt.grid(True)
            plt.legend()
            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())
            plt.savefig(f'experiments_{fig_type}.png')
    plt.show()


#############################################


def train_model(train_dataloader: torch.utils.data.DataLoader,
                val_dataloader: torch.utils.data.DataLoader,
                model: torch.nn.Module,
                config: Dict):
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    # Setting the training criterion and optimizer
    loss = torch.nn.CrossEntropyLoss().cuda()

    if config['Optimizer'] == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=config['Learning Rate'], weight_decay=config['Weight Decay'])
    elif config['Optimizer'] == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr=config['Learning Rate'],
                                momentum=0.9,
                                weight_decay=config['Weight Decay'])
    else:
        raise ValueError

    # Setting the model to train mode and loading it to the GPU
    model = model.cuda()

    best_val_epoch_accuracy = None
    best_model_filename = None

    for epoch in range(config['Num Epochs']):
        epoch_loss = 0
        epoch_accuracy = 0

        # Setting the model to train mode
        model.train()

        # iterate over the data

        for batch_idx, minibatch in enumerate(train_dataloader):
            x, labels = minibatch
            x = x.cuda()
            labels = labels.cuda()

            # Resetting the optimizer's gradients
            optim.zero_grad()

            # Forward Pass
            res = model(x)

            # Calculate the CE loss
            criteria = loss(res, labels)
            epoch_loss += criteria.item()

            # Calculate accuracy
            preds = res.argmax(dim=1, keepdim=True).squeeze()
            epoch_accuracy += torch.sum(preds.view(-1) == labels).item() / labels.shape[0]

            # Backpropagation
            criteria.backward()
            optim.step()

        # Run model evaluation on the validation set
        val_epoch_loss, val_epoch_accuracy = test_model(val_dataloader, model)
        print(f'Validation - Epoch {epoch}: Loss={val_epoch_loss}, Accuracy={val_epoch_accuracy}')

        # Saving the best model
        if best_val_epoch_accuracy is None or val_epoch_accuracy > best_val_epoch_accuracy:
            filename = join('./best_models',
                            'model_{}_lr{}_bs{}_epoch_{}.pth'.format(type(model).__name__.lower(),
                                                                     config["Learning Rate"],
                                                                     config["Batch Size"],
                                                                     epoch)
                            )
            torch.save(model.state_dict(), filename)
            best_val_epoch_accuracy = val_epoch_accuracy

            # Removing the previous best model and replacing with the new one
            if best_model_filename is not None:
                os.remove(best_model_filename)
            best_model_filename = filename

        # Saving the epoch losses and accuracies
        train_loss.append(epoch_loss / (batch_idx + 1))
        train_accuracy.append(epoch_accuracy / (batch_idx + 1))
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    return train_loss, train_accuracy, val_loss, val_accuracy


def test_model(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module):
    val_loss = 0
    val_accuracy = 0
    num_val_samples = 0

    # Setting the validation criterion
    loss = torch.nn.CrossEntropyLoss().cuda()

    # Setting the model to test mode
    model.eval()

    # iterate over the data
    with torch.no_grad():
        for batch_idx, minibatch in enumerate(dataloader):
            x, labels = minibatch
            x = x.cuda()
            labels = labels.cuda()
            num_val_samples += x.shape[0]

            # Forward Pass
            res = model(x)

            # Calculate the CE loss
            criteria = loss(res, labels)
            val_loss += criteria.item()

            # Calculate accuracy
            preds = res.argmax(dim=1, keepdim=True).squeeze()
            val_accuracy += torch.sum(preds == labels).item()

    val_accuracy /= num_val_samples
    val_loss /= (batch_idx + 1)

    return val_loss, val_accuracy


def main():
    config = {
        'Image Size': 64,
        'Batch Size': 128,
        'Num Epochs': 100,
        'Train Val Ratio': 0.9,
        'Num Workers': 8,
        'Learning Rate': 1e-3,
        'Weight Decay': 1e-4,
        'Optimizer': 'Adam',
        'Hidden Dims': [512, 256, 128],
        'Dropout Rate': 0.2,
        'Freeze Backbone': False,
        'Backbone': 'MobileNetV2'
    }

    trainset, testset = get_stl_datasets('./data/STL10')

    ##########################################################
    # Part 1 - Visualize the Data
    ##########################################################
    visualize_data(trainset)

    ##########################################################
    # Preprocessing the data
    ##########################################################
    # Splitting the training data into train and validation sets
    num_train_samples = int(trainset.data.shape[0] * config['Train Val Ratio'])
    num_val_samples = int(trainset.data.shape[0] - num_train_samples)
    train_set, val_set = torch.utils.data.random_split(trainset, [num_train_samples, num_val_samples])

    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=config['Batch Size'],
                                                   shuffle=True,
                                                   num_workers=config['Num Workers'])
    val_dataloader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=config['Batch Size'],
                                                 shuffle=True,
                                                 num_workers=config['Num Workers'])
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=config['Batch Size'], shuffle=False)

    # ##########################################################
    # Part 2 - Classification with Various Networks
    # ##########################################################
    experiments = {}
    batch_sizes = [8, 16, 32, 64, 128, 256]
    lr = [1e-2, 1e-3, 1e-4, 1e-5]
    hidden_dims = [[512, 256, 128],
                   [512, 512, 256, 256, 128],
                   [1024, 1024, 512, 512, 256, 256, 128]]
    dropout_rates = [0.0, 0.2, 0.5]
    optimizer = ['Adam', 'SGD']
    freeze_backbone = [True, False]

    exp_name = 'Learning Rate'
    for idx, param in enumerate(lr):
        # Creating a model context with the selected configuration
        model = get_model(model_type='CNN',
                          input_size=config['Image Size'] * config['Image Size'] * 3,
                          output_dim=NUM_CLASSES,
                          config=config)

        # Updating the value of the selected parameter
        config[exp_name] = param

        # Running the experiment
        train_loss, train_accuracy, val_loss, val_accuracy = train_model(train_dataloader=train_dataloader,
                                                                         val_dataloader=val_dataloader,
                                                                         model=model,
                                                                         config=config)

        experiments[param] = {f"Train Loss ({exp_name}={param})": train_loss,
                              f"Train Accuracy ({exp_name}={param})": train_accuracy,
                              f"Validation Loss ({exp_name}={param})": val_loss,
                              f"Validation Accuracy ({exp_name}={param})": val_accuracy}
        params_str = f'lr={config["Learning Rate"]}, batch={config["Batch Size"]}, optimizer={config["Optimizer"]}'
        model_type_str = type(model).__name__
        plot_losses_and_accuracies(train_loss, train_accuracy, val_loss, val_accuracy, model_type_str, params_str)

    # ##########################################################
    # # Part 3 - Models Evaluation
    # ##########################################################
    # model = get_model(model_type='CNN',
    #                   input_size=config['Image Size'] * config['Image Size'] * 3,
    #                   output_dim=NUM_CLASSES,
    #                   config=config).cuda()
    # filename = '/home/dev/Documents/PhD/deep.learning/ex3/best_models/model_cnn_lr0.001_bs128_mobilenetv2_epoch_40.pth'
    # model.load_state_dict(torch.load(filename))
    # test_loss, test_accuracy = test_model(test_dataloader, model)
    # print('Test - {}: Loss={:.3f}, Accuracy={:.3f}'.format(type(model).__name__, test_loss, test_accuracy))


if __name__ == "__main__":
    main()
