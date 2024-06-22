import os
import torchvision
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import pandas as pd
from data_loader.data_augmentation import GaussianBlur


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class CIFAR10_Imbalanced(object):
    def __init__(self, distributed=False, **options):
        if options['img_size']>=40:
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])

            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(options['img_size']),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ])
        
        else:
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.2023, 0.1994, 0.2010])
            transform_train = transforms.Compose([
                transforms.RandomCrop(options['img_size'], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize(options['img_size']),
            transforms.ToTensor(),
            normalize])
        
        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'cifar10')

        pin_memory = True if options['use_gpu'] else False

        trainset = IMBALANCECIFAR10(data_root, train=True, download=True, 
                                    transform=transform_train, 
                                    imb_type=options['imb_type'], 
                                    imb_factor=options['imb_factor'])

        
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        
        if distributed:
            sampler_train = DistributedSampler(trainset)
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_test = DistributedSampler(testset)
            testloader = DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_test,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

        else:
            trainloader = DataLoader(
                trainset, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=True,
            )
            
            
            testloader = DataLoader(
                testset, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=True,
            )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader

class CIFAR100_Imbalanced(object):
    def __init__(self, distributed=False, **options):
        if options['img_size']>=40:
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(options['img_size']),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ])
        
        else:
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.2023, 0.1994, 0.2010])

            transform_train = transforms.Compose([
                transforms.RandomCrop(options['img_size'], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize(options['img_size']),
            transforms.ToTensor(),
            normalize])
        

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'cifar100')
        trainset = IMBALANCECIFAR100(data_root, train=True, download=True, 
                                    transform=transform_train, 
                                    imb_type=options['imb_type'], 
                                    imb_factor=options['imb_factor'])

        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
        
        if distributed:
            sampler_train = DistributedSampler(trainset)
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_test = DistributedSampler(testset)
            testloader = DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_test,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

        else:
            trainloader = DataLoader(
                trainset, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=True,
            )
            
            
            testloader = DataLoader(
                testset, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=True,
            )
        
        self.num_classes = 100
        self.trainloader = trainloader
        self.testloader = testloader


class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        file_name = os.path.join(root, txt)
        with open(file_name) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.targets[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

class iNaturalist(object):
    def __init__(self, distributed=False, **options):
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380],
                                        std=[0.195, 0.194, 0.192])
        
        batch_size = options['batch_size']
        if options['aug_plus']==True:
            #moco v2
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(options['img_size'], scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            #moco v1
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(options['img_size'], scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize(options['img_size']+32),
            transforms.CenterCrop(options['img_size']),
            transforms.ToTensor(),
            normalize,
        ])

        train_set = LT_Dataset(root=options['dataroot'], txt='iNaturalist18_train.txt', transform=transform_train)
        val_set = LT_Dataset(root=options['dataroot'], txt='iNaturalist18_val.txt', transform=transform_test)
        
        if distributed:
            sampler_train = DistributedSampler(train_set)
            trainloader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_val = DistributedSampler(val_set)
            testloader = DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_val,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )


        else:
            trainloader = DataLoader(
                train_set, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=True,
            )

            valloader = DataLoader(
                val_set, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=True,
            )
            
            
            
        self.num_classes = 8142
        self.trainloader = trainloader
        self.valloader = valloader


class ImageNet(object):
    def __init__(self, distributed=False, **options):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        
        batch_size = options['batch_size']
        if options['aug_plus']==True:
            #moco v2
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(options['img_size'], scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            #moco v1
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(options['img_size']),
                #transforms.RandomGrayscale(p=0.2),
                #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize(options['img_size']+32),
            transforms.CenterCrop(options['img_size']),
            transforms.ToTensor(),
            normalize,
        ])
        
        train_set = LT_Dataset(root=options['dataroot'], txt='ImageNet_LT_train.txt', transform=transform_train)
        val_set = LT_Dataset(root=options['dataroot'], txt='ImageNet_LT_val.txt', transform=transform_test)
        test_set = LT_Dataset(root=options['dataroot'], txt='ImageNet_LT_test.txt', transform=transform_test)

        if distributed:
            sampler_train = DistributedSampler(train_set)
            trainloader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_train,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            sampler_val = DistributedSampler(val_set)
            valloader = DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_val,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

            sampler_test = DistributedSampler(test_set)
            testloader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_test,
                num_workers=options['num_workers'],
                pin_memory=True,
                drop_last=False
            )

        else:
            trainloader = DataLoader(
                train_set, 
                batch_size=batch_size, shuffle=True,
                num_workers=options['num_workers'], pin_memory=True,
            )

            valloader = DataLoader(
                val_set, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=True,
            )
            
            testloader = DataLoader(
                test_set, 
                batch_size=batch_size, shuffle=False,
                num_workers=options['num_workers'], pin_memory=True,
            )
            
        self.num_classes = 1000
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

