import argparse
import torch, torchvision
from torchvision import transforms
import GAN
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
from os import mkdir
from os.path import exists, join
import numpy as np
from typing import List

# Constants
SAMPLE_INTERVAL = 10
OUTPUT_FOLDER = './output'


def plot_loses(train_loss_g: List,
               train_loss_d: List,
               test_loss_g: List,
               test_loss_d: List,
               dataset_name: str,
               loss_name: str):
    # Retrieving the number of samples in the given series
    steps = np.arange(len(train_loss_g))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(steps, train_loss_g, label='Train Loss - G')
    ax[0].plot(steps, test_loss_g, label='Test Loss - G')
    ax[0].set_title('Generator')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)

    ax[1].plot(steps, train_loss_d, label='Train Loss - D')
    ax[1].plot(steps, test_loss_d, label='Test Loss - D')
    ax[1].set_title('Discriminator')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].grid(True)

    fig.suptitle('Dataset: {}\nLoss type: {}\nGenerator and Discriminator - Train and Test Losses'.format(dataset_name,
                                                                                                          loss_name))
    plt.legend()
    plt.show()


def train(G: torch.nn.Module,
          D: torch.nn.Module,
          trainloader: torch.utils.data.dataloader.DataLoader,
          optimizer_G: torch.optim,
          optimizer_D: torch.optim,
          latent_dim: int,
          g_mode: str):
    """
    Performs a single training epoch of both the Generator (G) and Discriminator (G)
    :param G: Generator module
    :param D: Discriminator module
    :param trainloader: Training data loader context
    :param optimizer_G: Generator's optimizer
    :param optimizer_D: Discriminator's optimizer
    :param latent_dim: The dimension of the generator's noise input
    :param g_mode: Generator mode: 'original'/'standard'
    :return: Generator and Discriminator training epoch losses
    """
    # Setting the modules to training mode
    G.train()
    D.train()

    # TODO
    # Resetting the epoch losses
    epoch_loss_g = 0
    epoch_loss_d = 0
    adversarial_loss = torch.nn.BCELoss()

    for batch_index, minibatch in enumerate(trainloader):
        # Setting the inputs for the adversarial loss
        batch_size = minibatch[0].shape[0]
        ones_val = Variable(torch.Tensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda()
        zeros_val = Variable(torch.Tensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda()
        real_imgs = minibatch[0].cuda()

        ##############################################
        # Generator - Forward and backward passes
        ##############################################
        optimizer_G.zero_grad()

        # Generate a batch of fake images
        fake_images = G(torch.randn(batch_size, latent_dim))

        # The Generator's success criteria - generate image that looks real
        if g_mode == 'original':
            generator_criteria = adversarial_loss(1 - D(fake_images), zeros_val)
        else:
            generator_criteria = adversarial_loss(D(fake_images), ones_val)
        epoch_loss_g += generator_criteria.item()

        # Generator's backpropagation
        generator_criteria.backward()
        optimizer_G.step()

        ##############################################
        # Discriminator - Forward and backward passes
        ##############################################
        optimizer_D.zero_grad()

        # Generate a batch of fake images
        fake_images = G(torch.randn(batch_size, latent_dim))

        # The Discriminator's success criteria - correctly indentify real and fake images
        real_loss = adversarial_loss(D(real_imgs), ones_val)
        fake_loss = adversarial_loss(D(fake_images.detach()), zeros_val)
        discriminator_criteria = (real_loss + fake_loss) / 2
        epoch_loss_d += discriminator_criteria.item()

        # Discriminator's backpropagation
        discriminator_criteria.backward()
        optimizer_D.step()

    # Calculating the epoch's average losses
    epoch_loss_g /= (batch_index + 1)
    epoch_loss_d /= (batch_index + 1)
    return epoch_loss_g, epoch_loss_d


def test(G: torch.nn.Module,
         D: torch.nn.Module,
         testloader: torch.utils.data.dataloader.DataLoader,
         latent_dim: int,
         g_mode: str):
    """
    Testing the performance of both the Generator (G) and Discriminator (G) moules
    :param G: Generator module
    :param D: Discriminator module
    :param testloader: Testing data loader context
    :param latent_dim: The dimension of the generator's noise input
    :param g_mode: Generator mode: 'original'/'standard'
    :return: Generator and Discriminator testing epoch losses
    """
    # Setting the module to inference mode
    G.eval()
    D.eval()

    # Resetting the epoch losses
    epoch_loss_g = 0
    epoch_loss_d = 0

    # Defining the adversarial loss
    adversarial_loss = torch.nn.BCELoss()

    with torch.no_grad():
        #TODO
        for batch_index, minibatch in enumerate(testloader):
            # Setting the inputs for the adversarial loss
            batch_size = minibatch[0].shape[0]
            ones_val = Variable(torch.Tensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda()
            zeros_val = Variable(torch.Tensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda()
            real_images = minibatch[0].cuda()

            # Generate a batch of fake images
            fake_images = G(torch.randn(batch_size, latent_dim))

            # The Generator's success criteria - generate image that looks real
            if g_mode == 'original':
                generator_criteria = adversarial_loss(1 - D(fake_images), zeros_val)
            else:
                generator_criteria = adversarial_loss(D(fake_images), ones_val)
            epoch_loss_g += generator_criteria.item()

            # Generate a batch of fake images
            fake_images = G(torch.randn(batch_size, latent_dim))

            # The Discriminator's success criteria - correctly indentify real and fake images
            real_loss = adversarial_loss(D(real_images), ones_val)
            fake_loss = adversarial_loss(D(fake_images.detach()), zeros_val)
            discriminator_criteria = (real_loss + fake_loss) / 2
            epoch_loss_d += discriminator_criteria.item()

        # Calculating the epoch's average losses
        epoch_loss_g /= (batch_index + 1)
        epoch_loss_d /= (batch_index + 1)
    return epoch_loss_g, epoch_loss_d


def sample(G: torch.nn.Module, sample_size: int, latent_dim: int):
    """
    Generates samples using the given Generator module
    :param G: Generator module
    :param sample_size: Number of samples to generate
    :param latent_dim: The dimension of the generator's noise input
    :return: Output samples generated by the Generator module
    """
    # Setting the module to inference mode
    G.eval()

    # Generating the samples
    with torch.no_grad():
        #TODO
        samples = G(torch.randn(sample_size, latent_dim))
    return samples


def main(args):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                     train=True,
                                                     download=True,
                                                     transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False,
                                                    download=True,
                                                    transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    # Creating the output folder if needed
    output_folder = f'{OUTPUT_FOLDER}_{args.dataset}'
    if not exists(output_folder):
        mkdir(output_folder)

    # Creating a Generator and Discriminator contexts
    G = GAN.Generator(latent_dim=args.latent_dim, batch_size=args.batch_size).cuda()
    D = GAN.Discriminator().cuda()

    # Defining an optimizer for the modules
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)

    #TODO
    train_loss_g, test_loss_g = [], []
    train_loss_d, test_loss_d = [], []

    for epoch in range(args.epochs):
        # Training a single epoch
        epoch_train_loss_g, epoch_train_loss_d = train(G,
                                                       D,
                                                       trainloader,
                                                       optimizer_G,
                                                       optimizer_D,
                                                       args.latent_dim,
                                                       args.g_mode)

        # Testing the model at the current epoch
        epoch_test_loss_g, epoch_test_loss_d = test(G, D, testloader, args.latent_dim, args.g_mode)

        # Reporting the losses
        print('Epoch {} - Train/Test losses: Generator={:.3f}, {:.3f}, Discriminator={:.3f}, {:.3f}'.format(
            epoch,
            epoch_train_loss_g,
            epoch_test_loss_g,
            epoch_train_loss_d,
            epoch_test_loss_d))

        # Saving the epoch losses for both the Generator and Discriminator
        train_loss_g.append(epoch_train_loss_g)
        train_loss_d.append(epoch_train_loss_d)
        test_loss_g.append(epoch_test_loss_g)
        test_loss_d.append(epoch_test_loss_d)

        # Saving sampled images from the model at every 10 epochs
        if epoch % SAMPLE_INTERVAL == 0:
            samples = sample(G, args.sample_size, args.latent_dim).unsqueeze(1)
            torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                         join(output_folder, f'{filename}_epoch_{epoch}.png'))

    samples = sample(G, args.sample_size, args.latent_dim).unsqueeze(1)
    torchvision.utils.save_image(torchvision.utils.make_grid(samples), join(output_folder, f'{filename}_final.png'))

    # Plotting the Generator's and Discriminator's train and test losses
    plot_loses(train_loss_g, train_loss_d, test_loss_g, test_loss_d, dataset_name=args.dataset, loss_name=args.g_mode)
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='fashion-mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=100)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=2e-4)
    parser.add_argument('--g_mode',
                        help='generator mode: original/standard',
                        type=str,
                        default='standard')

    args = parser.parse_args()
    main(args)
