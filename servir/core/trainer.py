import os
import torch

from servir.core.recorder import Recorder   
from servir.utils.main_utils import print_log, weights_to_cpu


def save_checkpoint(method, epoch, checkpoint_fname='checkpoint.pth'):

    checkpoint = {
        'epoch': epoch + 1,
        'optimizer': method.model_optim.state_dict(),
        'state_dict': weights_to_cpu(method.model.state_dict()),
        'scheduler': method.scheduler.state_dict()}
    
    torch.save(checkpoint, checkpoint_fname)

def train(train_loader, vali_loader, method, config, para_dict_fpath, checkpoint_fname, log_step = 1):

    epoch = 0
    max_epochs = config['max_epoch']


    """Training loops of STL methods"""
    recorder = Recorder(verbose=True)
    num_updates = epoch * config['steps_per_epoch']

    return_loss = True
    if config['dataname'] == 'wa_IR':
        skip_frame_loss = True
    else:
        skip_frame_loss = False

    eta = 1.0  # PredRNN variants
    for epoch in range(epoch, max_epochs):

        num_updates, train_loss, eta = method.train_one_epoch(train_loader, epoch, num_updates, \
                                                            eta, return_loss, skip_frame_loss)
        

        if epoch % log_step == 0:
            cur_lr = method.current_lr()
            cur_lr = sum(cur_lr) / len(cur_lr)
            with torch.no_grad():
                #===A validation loop during training==
                vali_loss, _ = method.vali(vali_loader, gather_pred=False, skip_frame_loss=skip_frame_loss)
                #=======================================
            if config['rank'] == 0:
                print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}'.format(
                    epoch + 1, len(train_loader), cur_lr, train_loss, vali_loss))
                
                # update and save best model as the one with lowest validation loss
                update_model = recorder(vali_loss)

                if update_model:
                    torch.save(method.model.state_dict(), para_dict_fpath)

                save_checkpoint(method, epoch, checkpoint_fname=checkpoint_fname)




    

