method = 'ConvLSTM'
# reverse scheduled sampling
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
# scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002
# model
num_hidden = '128, 128, 128, 128'
filter_size = 5
stride = 1
patch_size = 4
layer_norm = 0
relu_last = True
# training
max_epoch = 100 
early_stop_epoch= 10 
max_iter = 0
no_display_method_info= False
loss= 'threshold_square_loss' #options are ['MSE','threshold_square_loss','threshold_quantile_loss', 'neg_exponential']
quantile = 0.8
threshold = 2/53.2
# train optimizer
opt = 'adam'
lr = 5e-4
sched = 'onecycle'
opt_eps= None
opt_betas= None
momentum= 0.9
weight_decay = 0
clip_grad= None
clip_mode= 'norm'
# dataset
batch_size = 32 
val_batch_size = 32 
dataname = 'wa_imerg'
channels = 1
in_seq_length = 12
out_seq_length = 12 
img_height = 64
img_width = 64
# system
use_gpu = True
DataParallel = False
distributed = False
num_workers = 4

