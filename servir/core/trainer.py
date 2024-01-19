import logging
import torch

from servir.core.recorder import Recorder   


def train(train_loader, vali_loader, method, config, log_step = 1):

    epoch = 0
    iter = 0
    inner_iter = 0


    max_epochs = config['max_epoch']
    max_iters = None if config['max_iter'] == 0 else config['max_iter']
    early_stop = config['early_stop_epoch']


    """Training loops of STL methods"""
    recorder = Recorder(verbose=True, early_stop_time=min(early_stop, max_epochs))
    num_updates = epoch * config['steps_per_epoch']
    early_stop = False
    
    eta = 1.0  # PredRNN variants
    for epoch in range(epoch, max_epochs):

        num_updates, loss_mean, eta = method.train_one_epoch(train_loader, epoch, num_updates, eta)
        iter += 1

        if epoch % log_step == 0:
            cur_lr = method.current_lr()
            cur_lr = sum(cur_lr) / len(cur_lr)
            with torch.no_grad():
                results, _ = method.vali_one_epoch(vali_loader)
                vali_loss = results['loss'].mean()

            if self._rank == 0:
                print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                    epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss))
                early_stop = recorder(vali_loss, self.method.model, self.path)
                self._save(name='latest')
        if self._use_gpu and self.args.empty_cache:
            torch.cuda.empty_cache()
        if epoch > self._early_stop and early_stop:  # early stop training
            print_log('Early stop training at f{} epoch'.format(epoch))

    if not check_dir(self.path):  # exit training when work_dir is removed
        assert False and "Exit training because work_dir is removed"
    best_model_path = osp.join(self.path, 'checkpoint.pth')
    self._load_from_state_dict(torch.load(best_model_path))
    time.sleep(1)  # wait for some hooks like loggers to finish
    self.call_hook('after_run')


    def vali():
        """A validation loop during training"""
        results, eval_log = method.vali_one_epoch(self, self.vali_loader)
  
        if self._rank == 0:
            print_log('val\t '+eval_log)
            if has_nni:
                nni.report_intermediate_result(results['mse'].mean())

        return results['loss'].mean()