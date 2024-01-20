import os
import torch

from servir.core.recorder import Recorder   
from servir.utils.main_utils import print_log

def train(train_loader, vali_loader, method, config, log_step = 1):

    epoch = 0
    iter = 0

    max_epochs = config['max_epoch']


    """Training loops of STL methods"""
    recorder = Recorder(verbose=True)
    num_updates = epoch * config['steps_per_epoch']

    
    eta = 1.0  # PredRNN variants
    for epoch in range(epoch, max_epochs):

        num_updates, train_loss, eta = method.train_one_epoch(train_loader, epoch, num_updates, eta)
        iter += 1

        if epoch % log_step == 0:
            cur_lr = method.current_lr()
            cur_lr = sum(cur_lr) / len(cur_lr)
            with torch.no_grad():
                #===A validation loop during training==
                vali_loss, _ = method.vali(vali_loader, gather_pred=False)
                #=======================================
            if config['rank'] == 0:
                print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}'.format(
                    epoch + 1, len(train_loader), cur_lr, train_loss, vali_loss))
                
                # update and save best model as the one with lowest validation loss
                recorder(vali_loss, method, epoch, config['work_dir'])


    best_model_path = os.path.join(config['work_dir'], 'checkpoint.pth')
   

    return best_model_path


    

