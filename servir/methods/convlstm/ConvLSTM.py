import numpy as np
import torch
import torch.nn as nn
import torchvision

import pytorch_lightning as L
from servir.losses import get_loss


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t):
        
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * g_t
        return h_new, c_new



class ConvLSTM_Model(nn.Module):
    """ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, 
                num_hiddens: tuple = (128,128,128,128),
                img_input_channels: int = 1,
                img_output_channels: int = 1,
                image_shape: tuple = (360, 516),
                patch_size: int = 6,
                filter_size: int = 5,
                stride: int = 1,
                layer_norm: bool = True,
                in_seq_length: int = 4,
                out_seq_length: int = 12,
                reverse_scheduled_sampling: int = 0,
                relu_last: bool=True,
                ):
        super(ConvLSTM_Model, self).__init__()

        self.img_input_channels = img_input_channels
        self.img_output_channels = img_output_channels

        self.H, self.W = image_shape[0], image_shape[1]
        self.patch_size = patch_size    

        self.frame_channel = patch_size* patch_size * img_input_channels
        self.output_channel = patch_size * patch_size * img_output_channels
        self.num_layers = len(num_hiddens)
        self.num_hiddens = num_hiddens

        cell_list = []

        self.height = self.H // patch_size
        self.width = self.W // patch_size

        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length

        self.reverse_scheduled_sampling = reverse_scheduled_sampling
        self.relu_last = relu_last
        
        self.space2depth = nn.PixelUnshuffle(patch_size)
        self.depth2space = nn.PixelShuffle(patch_size)

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hiddens[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, self.num_hiddens[i], self.height, self.width, filter_size,
                                       stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(self.num_hiddens[self.num_layers - 1], self.output_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        

    def forward(self, x, mask):


        frames = self.space2depth(x)
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([x.shape[0], self.num_hiddens[i], self.height, self.width])
            zeros = zeros.to(x)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.in_seq_length + self.out_seq_length - 1):

            if t < self.in_seq_length:
                net = frames[:, t]
            else:
                net = mask[:, t - self.in_seq_length] * frames[:, t] + \
                    (1 - mask[:, t - self.in_seq_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])

            if self.relu_last:
                x_gen = torch.relu(x_gen)

            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)
        y = self.depth2space(next_frames[:, -self.out_seq_length:])

        return y, next_frames


class ConvLSTM(L.LightningModule):
    """ConvLSTM

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    Notice: ConvLSTM requires `find_unused_parameters=True` for DDP training.
    """

    def __init__(self, 
            num_hiddens: tuple = (128,128,128,128),
            lr: float = 5e-4,
            patch_size: int = 6,
            in_seq_length: int = 4,
            out_seq_length: int = 12,
            img_channel: int = 1,
            loss: str = 'mse',
            eta: float = 1.0,
            ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = len(self.num_hiddens)
        self.lr = lr

        self.patch_size = patch_size
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.img_channel = img_channel
        
        self.model = ConvLSTM_Model(self.num_hiddens)
        self.space2depth = nn.PixelUnshuffle(patch_size)
        self.depth2space = nn.PixelShuffle(patch_size)

        self.criterion = get_loss(loss)
        self.l1_loss = get_loss('l1')

        self.global_iteration = 0
        self.eta  = eta

    
    def forward(self, x, mask):
        """Forward the model"""

        y, next_frames = self.model(x, mask)
        
        return y, next_frames
    
    def get_true_mask(self, images):
        batch_size = images.shape[0]
        img_channel = images.shape[2]
        height, width = images.shape[3] // self.patch_size, images.shape[4] // self.patch_size
        channels = self.patch_size ** 2 * img_channel

        mask_true = torch.zeros((batch_size, self.out_seq_length - 1, channels,height, width))  
        mask_true = mask_true.to(images)

        return mask_true
    
        
    def schedule_sampling(self, images,\
                        sampling_stop_iter: int = 50000,
                        sampling_changing_rate: float = 0.00002, 
                        ):
        
        batch_size = images.shape[0]
        img_channel = images.shape[2]
        height, width = images.shape[3] // self.patch_size, images.shape[4] // self.patch_size
        channels = self.patch_size ** 2 * img_channel


        if self.global_iteration < sampling_stop_iter:
            self.eta -= sampling_changing_rate
        else:
            self.eta = 0.0

        random_flip = np.random.random_sample(
            (batch_size, self.out_seq_length - 1))
        true_token = (random_flip < self.eta)

        ones = torch.ones((channels, height, width))
        ones = ones.to(images)

        zeros =torch.zeros((channels, height, width))
        zeros = zeros.to(images)

        real_input_flag = []
        for i in range(batch_size):
            real_input_flag_b = []
            for j in range(self.out_seq_length - 1):
                if true_token[i, j]:
                    real_input_flag_b.append(ones)
                else:
                    real_input_flag_b.append(zeros)
                
            real_input_flag.append(torch.stack(real_input_flag_b))
                
        real_input_flag = torch.stack(real_input_flag)

        return real_input_flag


    def training_step(self, batch, batch_idx):
        self.global_iteration += 1
        
        in_images, out_images = batch

        images = torch.cat([in_images, out_images], dim=1)

        input_mask = self.schedule_sampling(images)

        _, pred_frames = self(images, input_mask)

        true_frames = self.space2depth(images)[:, 1:, :, :, :]

        loss = self.l1_loss(true_frames, pred_frames)

        self.log_dict(
            {
                "train/frames_l1_loss": self.criterion(true_frames, pred_frames),
            },
            prog_bar=True,
        )
        
        return loss


    def validation_step(self, batch, batch_idx):

        in_images, out_images = batch

        mask_true = self.get_true_mask(in_images)

        images = torch.cat([in_images, out_images], dim=1)
        pred_images, pred_frames = self(images, mask_true)

        true_frames = self.space2depth(images)[:, 1:, :, :, :]

        self.log_dict(
            {
                "val/frames_mse_loss": self.criterion(true_frames, pred_frames),
                "val/frames_l1_loss": self.l1_loss(true_frames, pred_frames),
                "val/images_mse_loss": self.criterion(out_images, pred_images),
                "val/images_l1_loss": self.l1_loss(out_images, pred_images),
            },
            prog_bar=True,
        )

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches),
            },
        }
    


