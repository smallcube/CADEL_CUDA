import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from logger import Logger
import time
import numpy as np
import warnings
import pdb
from models.util.loss import *
from torch.distributions import normal
from models.utils import mixup_data, WarmupCosineSchedule
from training_functions import update_score_base


class model():
    def __init__(self, config, data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.data = data
        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])

        # Initialize model, optimizer & scheduler
        networks_args = config['networks']
        def_file = networks_args['def_file']
        model_args = networks_args['params']
        self.model = source_import(def_file).create_model(**model_args)
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        #optimizer
        optimizer_args = config['optim_params']
        #dataset_cofig = config['dataset']
        lr = optimizer_args['lr']   #*dataset_cofig['batch_size']/256.0
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, 
                          momentum=optimizer_args['momentum'], 
                          weight_decay=optimizer_args['weight_decay'])

        #scheduler
        if config['coslr']:
            print("===> Using coslr eta_min={}".format(config['endlr']))
            self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=self.training_opt['warmup_epoch'], t_total=self.training_opt['num_epochs'])

            
        else:
            print("===> Using multistepLR eta_min={}".format(config['endlr']))
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.training_opt['milestones'])

        self.init_weight()
        self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
        self.logger.log_cfg(self.config)

    def init_weight(self):
        #prepare for logits adjustment
        self.cls_num_list = [0] * self.training_opt['num_classes']
        for label in self.data['train'].dataset.targets:
            self.cls_num_list[label] += 1
        self.cls_pos_list = []
        for cidx in tqdm(range(len(self.cls_num_list))):
            class_pos = torch.where(torch.tensor(self.data['train'].dataset.targets) == cidx)[0]
            self.cls_pos_list.append(class_pos)

        cls_num = np.load(self.training_opt['num_dir'])['num_per_class']
        cls_num = torch.tensor(cls_num).view(1, -1).cuda()
        self.distribution_source = torch.log(cls_num / torch.sum(cls_num)).view(1, -1)
        self.distribution_target = np.log(1.0 / self.training_opt['num_classes'])
    

    def train(self):
        # When training the network
        time.sleep(0.25)
        best_acc = 0.0
        best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            self.epoch = epoch
            
            torch.cuda.empty_cache()

            #update LOL score
            update_score_base(self.data['train'], self.model, self.cls_num_list, self.cls_pos_list, batch_size=self.config['dataset']['batch_size'])
            self.model.train()

            # Iterate over dataset
            total_preds = []
            total_labels = []
            step = 0
            for data in tqdm(self.data['train']):
                step += 1

                inputs, labels = data[0].cuda(), data[1].cuda()
                y_b = None
                lam = 1.0
                mixup_para = self.training_opt['mixup']
                inputs, labels, y_b, lam = mixup_data(inputs, labels, alpha=mixup_para['alpha'])
                
                # If training, forward with loss, and no top 5 accuracy calculation
                self.outputs = self.model(inputs)
                weight1, weight2 = None, None
                self.loss = 0
                logits_backup = 0
                for i in range(1, len(self.outputs)):
                    loss_i, weight1, weight2 = ensemble_loss_v2(pred=self.outputs[i], target=labels, target2=y_b, lam=lam,
                                                                weight1=weight1, weight2=weight2,
                                                                bins=self.training_opt['bins'], 
                                                                gamma=self.training_opt['gamma'],
                                                                base_weight=self.training_opt['base_weight'],
                                                                tempture=1.0-1.0*self.epoch/end_epoch)
                    self.loss = self.loss + loss_i
                    logits_backup = logits_backup + self.outputs[i]
                del weight1, weight2

                #logits_backup /= len(self.outputs) - 1
                self.logits_ensemble = logits_backup


                #self.batch_loss(labels, y_b, lam)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                # Tracking predictions
                _, preds = torch.max(self.logits_ensemble, 1)
                total_preds.append(torch2numpy(preds))
                total_labels.append(torch2numpy(labels))

                # update the accumulated prob
                # Output minibatch training results
                if step % self.training_opt['display_step'] == 0:
                    minibatch_loss_total = self.loss.item()
                    minibatch_acc = mic_acc_cal(preds, labels)

                    lr_current = max([param_group['lr'] for param_group in self.optimizer.param_groups])

                    print_str = ['Epoch: [%d/%d]'
                                    % (epoch, self.training_opt['num_epochs']),
                                    'Step: %5d'
                                    % (step),
                                    'Loss: %.3f'
                                    % (self.loss.item()),
                                    # 'Minibatch_loss_performance: %.3f'
                                    # % (minibatch_loss_perf) if minibatch_loss_perf else '',
                                    'current_learning_rate: %0.5f'
                                    % (lr_current),
                                    'Minibatch_accuracy_micro: %.3f'
                                    % (minibatch_acc)]
                    print_write(print_str, self.log_file)

                    loss_info = {
                        'Epoch': epoch,
                        'Step': step,
                        'Total': minibatch_loss_total,
                    }

                    self.logger.log_loss(loss_info)
                del self.logits_ensemble, self.loss, inputs

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = self.eval_with_preds(total_preds, total_labels)
            rsls_eval = self.eval(phase='val')
            rsls.update(rsls_train)
            rsls.update(rsls_eval)

            
            # Log results
            self.logger.log_acc(rsls)

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1
                self.save_model(epoch, best_epoch, best_acc)
                
            print('===> Saving checkpoint')
            self.save_latest(epoch)

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            self.scheduler.step()
            #if self.criterion_optimizer:
            #    self.criterion_optimizer_scheduler.step()

            del self.logits
            torch.cuda.empty_cache()

        print()
        print('Training Complete.')

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        #self.save_latest(epoch, best_epoch, best_acc)

        # Test on the test set
        self.eval('test' if 'test' in self.data else 'val')
        print('Done')

    def eval_with_preds(self, preds, labels):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)

        # Calculate normal prediction accuracy
        rsl = {'train_all': 0., 'train_many': 0., 'train_median': 0., 'train_low': 0.}
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = shot_acc(normal_preds, normal_labels, self.data['train'])
            rsl['train_all'] += len(normal_preds) / n_total * n_top1
            rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
            rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
            rsl['train_low'] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = \
                list(map(np.concatenate, [mixup_preds * 2, mixup_labels1 + mixup_labels2, mixup_ws]))
            mixup_ws = np.concatenate([mixup_ws, 1 - mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, self.data['train'])
            rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
            rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = ['\n Training acc Top1: %.3f \n' % (rsl['train_all']),
                     'Many_top1: %.3f' % (rsl['train_many']),
                     'Median_top1: %.3f' % (rsl['train_median']),
                     'Low_top1: %.3f' % (rsl['train_low']),
                     '\n']
        print_write(print_str, self.log_file)

        return rsl

    def eval(self, phase='val', tao=1.0, post_hoc=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        
        torch.cuda.empty_cache()

        self.model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        # Iterate over dataset
        for data in tqdm(self.data[phase]):
            inputs, labels = data[0].cuda(), data[1].cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                self.outputs = self.model(inputs)
                logits_backup = 0
                for i in range(0, len(self.outputs)):
                    if post_hoc:
                        logits_backup = logits_backup + self.outputs[i] + tao * (self.distribution_target - self.distribution_source)
                    else:
                        logits_backup = logits_backup + self.outputs[i]
                logits_backup /= len(self.outputs) - 1
                self.logits = logits_backup

                
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))

        
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1 = mic_acc_cal(preds[self.total_labels != -1],
                                             self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=False,
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1, \
        self.cls_accs = shot_acc(preds[self.total_labels != -1],
                                 self.total_labels[self.total_labels != -1],
                                 self.data['train'],
                                 acc_per_cls=True)
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s'
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f'
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f'
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f'
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f'
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f'
                     % (self.low_acc_top1),
                     '\n']

        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure}

        if phase == 'val':
            print_write(print_str, self.log_file)
        else:
            acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.many_acc_top1 * 100,
                self.median_acc_top1 * 100,
                self.low_acc_top1 * 100,
                self.eval_acc_mic_top1 * 100)]
            if self.log_file is not None and os.path.exists(self.log_file):
                print_write(print_str, self.log_file)
                print_write(acc_str, self.log_file)
            else:
                print(*print_str)
                print(*acc_str)

        if phase == 'test':
            with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
                pickle.dump(self.cls_accs, f)
        return rsl

    
    
    def save_latest(self, epoch):
        model_states = {
            'epoch': epoch,
            'state_dict': self.model.state_dict()
        }

        model_dir = os.path.join(self.training_opt['log_dir'],
                                 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)

    def save_model(self, epoch, best_epoch, best_acc):

        model_states = {'epoch': epoch,
                        'best_epoch': best_epoch,
                        'state_dict_best': self.model.state_dict(),
                        'best_acc': best_acc}

        model_dir = os.path.join(self.training_opt['log_dir'],
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)

    