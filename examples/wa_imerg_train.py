import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning import Trainer
from servir.datasets.dataLoader_imerg_from_tif import WAImergDataModule

from servir.methods.convlstm.ConvLSTM import ConvLSTM
from servir.utils import create_folder


#================Specification=========================#
method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
# file names
result_path = os.path.join(base_path, 'results', dataset_name, method_name)
create_folder(result_path, level=3)

# loss to use and normalization method of data
loss = 'l1'
normalize_method ='01range'

# set relu_last to True when using 01range or no normalization, as all pixel values are positive.
# set relu_last to False when using mean_std normalization, as pixel values can be negative.
model = ConvLSTM(loss=loss, layer_norm= True, relu_last=True)

early_stop_callback = EarlyStopping(monitor="val/frames_l1_loss", min_delta=0.00, patience=2, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(monitor='val/frames_l1_loss', dirpath=result_path, filename=f'loss:{loss}--{normalize_method}')# '{epoch:02d}-{val_loss:.2f}'


# data module
train_st = '2020-08-30 00:00:00' 
train_ed = '2020-08-31 23:30:00' 
val_st = '2020-09-30 00:00:00'
val_ed = '2020-09-30 23:30:00'
test_st = '2020-10-31 00:00:00' 
test_ed = '2020-10-31 23:30:00'



# get the data module
dataPath = os.path.join(base_path, 'data', dataset_name)
data_module = WAImergDataModule(dataPath, train_st, train_ed, val_st, val_ed, \
                                in_seq_length=4, out_seq_length=12, normalize_method=normalize_method,\
                                img_shape = (360, 516))


trainer = Trainer(
    max_epochs=10,
    callbacks=[early_stop_callback, checkpoint_callback],
    accelerator="gpu",
    devices=4, 
    strategy="ddp", 
    num_nodes=4
)

trainer.fit(model, data_module)







