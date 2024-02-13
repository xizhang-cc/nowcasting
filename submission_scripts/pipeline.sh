

python m_tif2h5py.py \
'/vol_efthymios/NFS07/en279/SERVIR/temp/' \
'/vol_efthymios/NFS07/en279/SERVIR/temp/input_imerg.h5'

python m_nowcasting.py \
'/vol_efthymios/NFS07/en279/SERVIR/nowcasting' \
'/vol_efthymios/NFS07/en279/SERVIR/temp/ConvLSTM_Config.py' \
'/vol_efthymios/NFS07/en279/SERVIR/temp/imerg_only_mse_params.pth' \
False \
'/vol_efthymios/NFS07/en279/SERVIR/temp/input_imerg.h5' \
'/vol_efthymios/NFS07/en279/SERVIR/temp/output_imerg.h5'

python m_h5py2tif.py '
/vol_efthymios/NFS07/en279/SERVIR/temp/output_imerg.h5' \
'/vol_efthymios/NFS07/en279/SERVIR/temp/imerg_giotiff_meta.json' \
'/vol_efthymios/NFS07/en279/SERVIR/temp/'

