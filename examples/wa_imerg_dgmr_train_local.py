
import os
import datetime

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

from servir.datasets.dataLoader_imerg_from_npy import WAImergNpyDataRSModule
from servir.methods.dgmr.dgmr import DGMR
from servir.utils import create_folder

def main():
    #================Specification=========================#
    method_name = 'dgmr'
    dataset_name = 'wa_imerg'

    base_path = '/home/cc/projects/nowcasting' 
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
    img_shape = (360, 516)
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

    early_stop_callback = EarlyStopping(monitor="val/frames_l1_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val/frames_l1_loss', dirpath=result_path, filename=best_model_fname)# '{epoch:02d}-{val_loss:.2f}'


    # data module
    train_st = '2019-12-22 00:00:00' 
    train_ed = '2019-12-31 23:30:00' 

    # get the data module
    dataPath = os.path.join(base_path, 'data', dataset_name)
    data_module = WAImergNpyDataRSModule(dataPath, train_st, train_ed, sampling_freq=sampling_freq,\
                                        in_seq_length=in_seq_length, out_seq_length=out_seq_length, normalize_method=normalize_method,\
                                        img_shape = img_shape, batch_size=batch_size)

    trainer = Trainer(
        max_epochs=100,
        callbacks=[early_stop_callback, checkpoint_callback],
        precision=32,
        accelerator="gpu",
    )

    trainer.fit(model, data_module,
                # ckpt_path='last',
                )

if __name__ == "__main__":
    main()


