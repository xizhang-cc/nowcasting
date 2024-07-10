import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning import Trainer
from servir.datasets.dataLoader_imerg_from_tif import WAImergDataModule

from servir.methods.convlstm.ConvLSTM import ConvLSTM
from servir.utils.utils import create_folder


#================Specification=========================#
method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#


train_st = '2020-08-30 00:00:00' 
train_ed = '2020-08-31 23:30:00' 
val_st = '2020-09-30 00:00:00'
val_ed = '2020-09-30 23:30:00'
test_st = '2020-10-31 00:00:00' 
test_ed = '2020-10-31 23:30:00'

normalize_method = None#'01range'


# file names
result_path = os.path.join(base_path, 'results', dataset_name, method_name)
create_folder(result_path, level=3)



# get the data module
dataPath = os.path.join(base_path, 'data', dataset_name)
data_module = WAImergDataModule(dataPath, train_st, train_ed, val_st, val_ed, test_st, test_ed, \
                                in_seq_length=4, out_seq_length=12, normalize_method=normalize_method,\
                                precip_mean=0.0, precip_std=1.0, precip_max=1.0, precip_min=0.0, img_shape = (360, 516))

# load the model
model = ConvLSTM()

early_stop_callback = EarlyStopping(monitor="val/frames_l1_loss", min_delta=0.00, patience=2, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(monitor='val/frames_l1_loss', dirpath=result_path, filename=f'l1_loss_{normalize_method}')# '{epoch:02d}-{val_loss:.2f}'

trainer = Trainer(
    max_epochs=10,
    callbacks=[early_stop_callback, checkpoint_callback],
    accelerator="auto",
)

trainer.fit(model, data_module)







