
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from servir.datasets.dataLoader_imerg_from_h5 import ghanaImergDataModule
from servir.methods.dgmr.dgmr import DGMR
from servir.utils import create_folder

def main():
    #================Specification=========================#
    method_name = 'dgmr'
    dataset_name = 'ghana_imerg'

    base_path = "/home1/zhang2012/nowcasting/"
    # file names
    result_path = os.path.join(base_path, 'results', dataset_name, method_name)
    create_folder(result_path, level=3)

    # loss to use and normalization method of data
    loss = 'l1'
    normalize_method ='01range'
    best_model_fname = f'{method_name}-{loss}-{normalize_method}' # no need to add .ckpt extension

    batch_size = 12
    in_seq_length = 4
    out_seq_length = 12
    img_shape = (64, 64)

    model = DGMR(
            forecast_steps=out_seq_length,
            input_channels=1,
            output_shape=img_shape,
            latent_channels=384,
            context_channels=192,
            generation_steps = 6, # number of generation steps
            num_input_frames = in_seq_length, # number of input frames
            num_layers = 2, # number of layers in the temporal and spatial encoder
        )

    checkpoint_callback = ModelCheckpoint(monitor="train/g_loss", dirpath=result_path, filename=best_model_fname, save_last=True)# '{epoch:02d}-{val_loss:.2f}'

    # data module
    train_start_date = '2011-10-01 00:00:00'
    train_end_date = '2018-10-31 23:30:00'
    val_start_date = '2019-10-01 00:00:00'
    val_end_date = '2019-10-31 23:30:00'


    # get the data module
    dataPath = os.path.join(base_path, 'data', dataset_name)
    data_module = ghanaImergDataModule(os.path.join(dataPath, 'ghana_imerg_2011_2020_oct.h5'), \
                                    train_start_date, train_end_date, val_start_date, val_end_date,\
                                    in_seq_length=in_seq_length, out_seq_length=out_seq_length, \
                                    normalize_method=normalize_method, batch_size=batch_size)

    trainer = Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback],
        precision=32,
        accelerator="gpu",
        devices=4, 
        strategy="ddp_find_unused_parameters_true", 
        num_nodes=2,
    )

    trainer.fit(model, data_module,
                # ckpt_path='last',
                )

if __name__ == "__main__":
    main()


