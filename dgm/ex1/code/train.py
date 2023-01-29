"""Training procedure for NICE.
"""
import argparse
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import nice
from os import getcwd, mkdir
from os.path import join, exists


def train(flow: torch.nn.Module,
          trainloader: torch.utils.data.DataLoader,
          optimizer: torch.optim) -> float:
    # set to training mode
    flow.train()

    # Setting an accumulated epoch loss
    running_loss = 0
    num_iterations = 0

    # Going over the training dataset (single epoch)
    for inputs, _ in trainloader:
        # change  shape from BxCxHxW to Bx(C*H*W)
        inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]).to(flow.device)

        # Model forward pass
        model_res = flow(inputs)

        # Calculating the loss criteria
        optimizer.zero_grad()
        loss = -model_res.mean()
        running_loss += loss
        num_iterations += 1

        # Backprop and optimization
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / num_iterations
    return epoch_loss.item()


def test(flow: torch.nn.Module,
         testloader: torch.utils.data.DataLoader,
         filename: str,
         epoch: int,
         sample_shape: int) -> float:
    # set to inference mode
    flow.eval()

    # Setting an accumulated test loss
    running_loss = 0
    num_iterations = 0

    with torch.no_grad():
        samples = flow.sample(100).cpu()
        a, b = samples.min(), samples.max()
        samples = (samples - a) / (b - a + 1e-10)
        samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples), filename + 'epoch%d.png' % epoch)

        # Going over the test dataset (single epoch)
        for inputs, _ in testloader:
            # change  shape from BxCxHxW to Bx(C*H*W)
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[3]).to(flow.device)

            # Model forward pass
            model_res = flow(inputs)

            # Calculating the loss criteria
            loss = -model_res.mean()
            running_loss += loss
            num_iterations += 1

    test_loss = running_loss / num_iterations
    return test_loss.item()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sample_shape = [1, 28, 28]
    full_dim = 1 * 28 * 28

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.))  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%d_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden \
                          + '.pt'

    # Creating an output folder for the selected dataset
    dataset_samples_folder = f'{args.dataset}_samples'
    if not exists(dataset_samples_folder):
        mkdir(dataset_samples_folder)
    dataset_samples_folder = join(dataset_samples_folder, args.coupling_type)
    if not exists(dataset_samples_folder):
        mkdir(dataset_samples_folder)

    # Creating the NICE model
    flow = nice.NICE(prior=args.prior,
                     coupling=args.coupling,
                     coupling_type=args.coupling_type,
                     in_out_dim=full_dim,
                     mid_dim=args.mid_dim,
                     hidden=args.hidden,
                     device=device).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=args.lr)

    train_loss = []
    test_loss = []
    for epoch in range(args.epochs):
        # Running training
        train_loss.append(train(flow, trainloader, optimizer))

        # Running test on single epoch
        test_loss.append(test(flow=flow,
                              testloader=testloader,
                              filename=join(dataset_samples_folder, f"{args.dataset}_sampled_"),
                              epoch=epoch,
                              sample_shape=sample_shape))

        # Printing the training and test losses' values
        print(f"Epoch {epoch}:  train loss={train_loss[-1]}, test loss={test_loss[-1]}")

        # Saving the model
        if epoch % 10 == 0:
            torch.save(flow.state_dict(), "./models/" + model_save_filename)

    # Ploting the train and test loss over the training process (for each epoch)
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.title("Train and Test Loss vs. Epochs\n" + f'Dataset: {args.dataset}, Coupling: {args.coupling_type}')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(join(getcwd(), f"loss_{args.dataset}_loss_coupling_{args.coupling_type}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', help='dataset to be modeled.', type=str, default='fashion-mnist')
    parser.add_argument('--prior', help='latent distribution.', type=str, default='logistic')
    parser.add_argument('--batch_size', help='number of images in a mini-batch.', type=int, default=128)
    parser.add_argument('--epochs', help='maximum number of iterations.', type=int, default=50)
    parser.add_argument('--sample_size', help='number of images to generate.', type=int, default=64)
    parser.add_argument('--coupling_type', help='.', type=str, default='affine')
    parser.add_argument('--coupling', help='.', type=int, default=4)
    parser.add_argument('--mid-dim', help='.', type=int, default=1000)
    parser.add_argument('--hidden', help='.', type=int, default=5)
    parser.add_argument('--lr', help='initial learning rate.', type=float, default=1e-3)
    input_args = parser.parse_args()
    main(input_args)
