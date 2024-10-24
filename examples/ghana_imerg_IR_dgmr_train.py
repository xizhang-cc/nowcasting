
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from servir.datasets.dataLoader_ghana_imerg_IR_h5 import ghanaImergIRRSDataModule
from servir.methods.dgmr.dgmr import DGMR
from servir.utils import create_folder

def main():
    #================Specification=========================#
    method_name = 'dgmr'
    dataset_name = 'ghana_imerg_IR'

    base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/" #
    # file names
    result_path = os.path.join(base_path, 'results', dataset_name, method_name)
    create_folder(result_path, level=3)

    # loss to use and normalization method of data
    normalize_method ='gaussian'
    IR_sparse = True
    best_model_fname = f'{dataset_name}-{method_name}-{normalize_method}' # no need to add .ckpt extension

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
            num_input_frames = in_seq_length*2, # number of input frames
            num_layers = 2, # number of layers in the temporal and spatial encoder
        )

    checkpoint_callback = ModelCheckpoint(monitor="train/g_loss", dirpath=result_path,\
                                        filename=best_model_fname, save_last=True)# '{epoch:02d}-{val_loss:.2f}'

    # data module
    train_start_date = '2011-10-01 00:00:00'
    train_end_date = '2019-10-31 23:30:00'

    train_val_split = [0.9, 0.1]


    # get the data module
    dataPath1 = os.path.join(base_path, 'data', 'ghana_imerg')
    dataPath2 = os.path.join(base_path, 'data', 'ghana_IR')

    fPath1 = os.path.join(dataPath1, 'ghana_imerg_2011_2020_oct.h5')
    fPath2 = os.path.join(dataPath2, 'ghana_IR_2011_2020_oct.h5')
    data_module = ghanaImergIRRSDataModule(fPath1, fPath2,\
                                        train_start_date, train_end_date, train_val_split=train_val_split,\
                                        in_seq_length=in_seq_length, out_seq_length=out_seq_length, \
                                        batch_size=batch_size, IR_sparse=IR_sparse)

    trainer = Trainer(
        max_epochs=1000,
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


