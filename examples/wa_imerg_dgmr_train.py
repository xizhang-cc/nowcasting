
import os
import datetime

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from servir.datasets.dataLoader_imerg_from_npy import WAImergNpyDataRSModule
from servir.methods.dgmr.dgmr import DGMR
from servir.utils import create_folder

def main():
    #================Specification=========================#
    method_name = 'dgmr'
    dataset_name = 'wa_imerg'

    base_path = "/home1/zhang2012/nowcasting/"
    # file names
    result_path = os.path.join(base_path, 'results', dataset_name, method_name)
    create_folder(result_path, level=3)

    # loss to use and normalization method of data
    loss = 'l1'
    normalize_method ='01range'
    best_model_fname = f'{method_name}-{loss}-{normalize_method}' # no need to add .ckpt extension

    batch_size = 2
    in_seq_length = 4
    out_seq_length = 12
    img_shape = (352, 512)
    sampling_freq = datetime.timedelta(hours=2)

    model = DGMR(
            forecast_steps=out_seq_length,
            input_channels=1,
            output_shape=img_shape,
            latent_channels=768,
            context_channels=384,
            num_samples=2,
            num_input_frames = in_seq_length,
        )

    checkpoint_callback = ModelCheckpoint(monitor="train/g_loss", dirpath=result_path, filename=best_model_fname, save_last=True)# '{epoch:02d}-{val_loss:.2f}'

    # data module
    # data module
    train_st = '2017-01-01 00:00:00' 
    train_ed = '2019-12-31 23:30:00' 


    # get the data module
    dataPath = os.path.join(base_path, 'data', dataset_name)
    data_module = WAImergNpyDataRSModule(dataPath, train_st, train_ed, sampling_freq=sampling_freq,\
                                        in_seq_length=in_seq_length, out_seq_length=out_seq_length, normalize_method=normalize_method,\
                                        img_shape = img_shape, batch_size=batch_size)

    trainer = Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback],
        precision=32,
        accelerator="gpu",
        devices=4, 
        strategy="ddp_find_unused_parameters_true", 
        num_nodes=4,
    )

    trainer.fit(model, data_module,
                # ckpt_path='last',
                )

if __name__ == "__main__":
    main()


