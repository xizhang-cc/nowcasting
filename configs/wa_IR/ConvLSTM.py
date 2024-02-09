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
num_hidden = '128,128,128,128' #
filter_size = 5
stride = 1
patch_size = 6
layer_norm = 0
# training
max_epoch = 100
early_stop_epoch= -1
max_iter = 0
no_display_method_info= False
skip_frame_loss=True
loss = 'MSE'    
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
batch_size = 2
val_batch_size = 2
dataname = 'wa_IR'
channels = 1
in_seq_length = 10
out_seq_length = 9
img_height = 360
img_width = 516
# system
use_gpu = True
DataParallel = False
distributed = False
num_workers = 4
