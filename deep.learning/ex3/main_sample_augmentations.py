import torch as torch
import matplotlib.pyplot as plt
import torchvision as torchvision
from torchvision import transforms
import numpy as np

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = transforms.Compose([#transforms.RandomCrop(64),
                                      transforms.RandomRotation(degrees=(0, 20),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                                      # transforms.ColorJitter(brightness=.1, saturation=.1, contrast=.1, hue=.1),
                                      # transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      # transforms.Normalize(mean, std),
                                      ])
image_index = 40
trainset = torchvision.datasets.STL10(root='./data/STL10', split='train', download=True, transform=train_transform)
augmentation_sampler = torch.utils.data.SubsetRandomSampler([image_index])
augmentation_loader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=4, sampler=augmentation_sampler)

for x, labels in augmentation_loader:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Augmentation: Random Horizontal Flip')
    ax[0].imshow(np.moveaxis(augmentation_loader.dataset.data[image_index], [0, 1, 2], [2, 0, 1]))
    ax[0].set_title('Input Image')
    ax[1].imshow(torch.permute(x[-1].squeeze(), (1, 2, 0)))
    ax[1].set_title('Augmented Image')
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.show()
    break
