import os
import datetime

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

from servir.datasets.dataLoader_imerg_from_npy import WAImergNpyDataModule
from servir.methods.convlstm.ConvLSTM import ConvLSTM
from servir.utils import create_folder

def main():
    #================Specification=========================#
    method_name = 'ConvLSTM'
    dataset_name = 'wa_imerg'

    base_path = "/home1/zhang2012/nowcasting/"#'/home/cc/projects/nowcasting' #
    # file names
    result_path = os.path.join(base_path, 'results', dataset_name, method_name)
    create_folder(result_path, level=3)

    # loss to use and normalization method of data
    loss = 'l1'
    normalize_method ='01range'
    batch_size =12
    best_model_fname = f'{method_name}-{loss}-{normalize_method}' # no need to add .ckpt extension


    # set relu_last to True when using 01range or no normalization, as all pixel values are positive.
    # set relu_last to False when using mean_std normalization, as pixel values can be negative.
    model = ConvLSTM(loss=loss, layer_norm= True, relu_last=True)

    early_stop_callback = EarlyStopping(monitor="val/frames_l1_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val/frames_l1_loss', dirpath=result_path, filename=best_model_fname, save_last=True)# '{epoch:02d}-{val_loss:.2f}'


    # data module
    train_st = '2017-01-01 00:00:00' 
    train_ed = '2018-12-31 23:30:00' 

    val_st = '2019-01-01 00:00:00' 
    val_ed = '2019-12-31 23:30:00' 

    # get the data module
    dataPath = os.path.join(base_path, 'data', dataset_name)
    data_module = WAImergNpyDataModule(dataPath, train_st, train_ed, val_st, val_ed,\
                                        sampling_freq=datetime.timedelta(minutes=30),\
                                        in_seq_length=4, out_seq_length=12, normalize_method=normalize_method,\
                                        img_shape = (360, 516), batch_size=batch_size)


    trainer = Trainer(
        max_epochs=100,
        callbacks=[early_stop_callback, checkpoint_callback],
        precision=32,
        accelerator="gpu",
        devices=4, 
        strategy="ddp_find_unused_parameters_true", 
        num_nodes=2
    )

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()


