import os
import numpy as np
import torch
from servir.utils.main_utils import  weights_to_cpu, print_log


class Recorder:
    def __init__(self, verbose=True, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        update_model = False

        score = -val_loss

        if (self.best_score is None) or (score >= self.best_score - self.delta):

            update_model = True

            if self.verbose:
                print_log(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). best model updated.\n')

            # update best score and minimum validation loss
            self.best_score = score
            self.val_loss_min = val_loss
        
        else:
            if self.verbose:
                print_log('Validation loss higher than current best model, best model not updated.\n')

        return update_model


        

        