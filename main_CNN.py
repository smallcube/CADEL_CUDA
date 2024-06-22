import os
import argparse
import pprint
from run_networks_CNN_DDP import model
import yaml
from utils import get_value
from data_loader.get_loader import get_dataloader
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__=='__main__':

    data_root = {'ImageNet': './data/ImageNet_LT/',
                'Places': './data/Places_LT/',
                'iNaturalist18': './data/iNaturalist18/'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config/ImageNet_LT/resnet50_stage1.yaml', type=str)
    
    args = parser.parse_args()

    # LOAD CONFIGURATIONS
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    
    training_opt = config['training_opt']
    #relatin_opt = config['memory']
    dataset = config['dataset']

    if not os.path.isdir(training_opt['log_dir']):
        os.makedirs(training_opt['log_dir'])

    pprint.pprint(config)

    data = get_dataloader(dataset)
    
    training_model = model(config, data)

    training_model.train()
            
    print('ALL COMPLETED.')
