import os
from data_loader.imbalance_data.lt_data import LT_Dataset
import torch

def get_dataloader(options=None):
    print("{} Preparation".format(options['dataset']))
    if 'ImageNet' in options['dataset']:
        train_data = LT_Dataset(root=options['root'], 
                                txt= os.path.join(options['root'], "ImageNet_LT_train.txt"), 
                                args=options, 
                                dataset='imgnet', 
                                loss_type='CE', 
                                use_randaug=options['use_randaug'], 
                                split='train', 
                                aug_prob=options['aug_prob'],
                                upgrade=options['cuda_upgrade'], 
                                downgrade=options['cuda_downgrade'])
        
        val_data = LT_Dataset(root=options['root'], 
                                txt= os.path.join(options['root'], "ImageNet_LT_val.txt"), 
                                args=options, 
                                dataset='imgnet', 
                                loss_type='CE', 
                                split='valid')
        
        test_data = LT_Dataset(root=options['root'], 
                                txt= os.path.join(options['root'], "ImageNet_LT_test.txt"), 
                                args=options, 
                                dataset='imgnet', 
                                loss_type='CE', 
                                split='valid')
        
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_data, 
                                                   batch_size=options['batch_size'], 
                                                shuffle=(train_sampler is None),
                                                num_workers=options['workers'], 
                                                pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(test_data, 
                                                batch_size=options['batch_size'], 
                                                shuffle=False,
                                                num_workers=options['workers'], 
                                                pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, 
                                                batch_size=options['batch_size'], 
                                                shuffle=False,
                                                num_workers=options['workers'], 
                                                pin_memory=True)
        
    dataloader = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloader