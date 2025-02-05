from PIL import Image
import torch
from torch.utils.data import Sampler
from torchvision import transforms, datasets

from dataloader.dataset.TinyImageNet import TinyImageNet, Custom_TinyImageNet
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
class Custom_CIFAR100(CIFAR100):
    # ------------------------
    # Custom CIFAR-100 dataset which returns returns 1 images, 1 target, image index
    # ------------------------
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, [target, index]


class Custom_CIFAR10(CIFAR10):
    # ------------------------
    # Custom CIFAR-10 dataset which returns returns 1 images, 1 target, image index
    # ------------------------
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, [target, index]



class Custom_ImageFolder(ImageFolder):
    #------------------------
    #Custom ImageFolder dataset which returns 1 images, 1 target, image index
    #------------------------
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, [target, index]


def load_dataset(args):
    name = args.dataset
    root = args.data
    if name.startswith('cifar'):
        root = root + '/' + name

        transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ]

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize([0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761]))

        transform_train = transforms.Compose(transforms_list)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        if name.endswith('100'):
            if args.method == 'Smooth':
                trainset = Custom_CIFAR100(root, train=True,  download=True, transform=transform_train)
            else:
                trainset = datasets.CIFAR100(root, train=True,  download=True, transform=transform_train)

            valset   = datasets.CIFAR100(root, train=False, download=True, transform=transform_test)
        else:
            if args.method == 'Smooth':
                trainset = Custom_CIFAR10(root, train=True, download=True, transform=transform_train)
            else:
                trainset = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)

            valset = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)

    elif name =='tinyimagenet':
        transforms_list = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)))

        transform_train = transforms.Compose(transforms_list)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        ])

        if args.method == 'Smooth':
            trainset = Custom_TinyImageNet(root, transform=transform_train)
        else:
            trainset = TinyImageNet(root, transform=transform_train)

        valset = TinyImageNet(root,download=True, split='val', transform=transform_test, target_transform=None)

    else:
        raise Exception('Unknown dataset: {}'.format(name))

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=(torch.cuda.is_available()))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=(torch.cuda.is_available()))

    return trainloader, valloader
