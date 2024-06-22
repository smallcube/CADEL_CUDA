from data_loader.data_loader import *


def get_dataloader(distributed=False, options=None):
    print("{} Preparation".format(options['dataset']))
    testloader=None
    if 'cifar10_lt' == options['dataset'].lower():
        Data = CIFAR10_Imbalanced(distributed=distributed, **options)
        trainloader, valloader = Data.trainloader, Data.valloader
    elif 'cifar100_lt' == options['dataset'].lower():
        Data = CIFAR100_Imbalanced(distributed=distributed, **options)
        trainloader, valloader = Data.trainloader, Data.valloader
    elif 'imagenet_lt' == options['dataset'].lower():
        Data = ImageNet(distributed=distributed, **options)
        trainloader, valloader, testloader = Data.trainloader, Data.valloader, Data.testloader
    elif 'inaturalist' == options['dataset'].lower():
        Data = iNaturalist(distributed=distributed, **options)
        trainloader, valloader = Data.trainloader, Data.valloader
    
    if testloader is None:
        testloader = valloader

    dataloader = {'train': trainloader, 'val': valloader, 'test': testloader}
    return dataloader