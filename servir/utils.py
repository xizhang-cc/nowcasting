
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly

from pysteps.datasets import  create_default_pystepsrc

    
#===============================================================================
#===========================Load GeoTiff format data============================
# =============================================================================== 
def ReadandWarp(gridFile,xmin,ymin,xmax,ymax):
    #Read grid and warp to domain grid
    #Assumes no reprojection is necessary, and EPSG:4326
    rawGridIn = gdal.Open(gridFile, GA_ReadOnly)
    # Adjust grid
    pre_ds = gdal.Translate('OutTemp.tif', rawGridIn, options="-co COMPRESS=Deflate -a_nodata 29999 -a_ullr -180.0 90.0 180.0 -90.0")

    gt = pre_ds.GetGeoTransform()
    proj = pre_ds.GetProjection()
    nx = pre_ds.GetRasterBand(1).XSize
    ny = pre_ds.GetRasterBand(1).YSize
    NoData = 29999
    pixel_size = gt[1]

    #Warp to model resolution and domain extents
    ds = gdal.Warp('', pre_ds, srcNodata=NoData, srcSRS='EPSG:4326', dstSRS='EPSG:4326', dstNodata='-9999', format='VRT', xRes=pixel_size, yRes=-pixel_size, outputBounds=(xmin,ymin,xmax,ymax))

    WarpedGrid = ds.ReadAsArray()
    new_gt = ds.GetGeoTransform()
    new_proj = ds.GetProjection()
    new_nx = ds.GetRasterBand(1).XSize
    new_ny = ds.GetRasterBand(1).YSize

    return WarpedGrid, new_nx, new_ny, new_gt, new_proj

def WriteGrid(gridOutName, dataOut, nx, ny, gt, proj):
    #Writes out a GeoTIFF based on georeference information in RefInfo
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(gridOutName, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dataOut.shape = (-1, nx)
    dst_ds.GetRasterBand(1).WriteArray(dataOut, 0, 0)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    dst_ds = None

def processIMERG(local_filename,llx,lly,urx,ury):
  # Process grid
  # Read and subset grid
  NewGrid, nx, ny, gt, proj = ReadandWarp(local_filename,llx,lly,urx,ury)

  # Scale value
  NewGrid = NewGrid*0.1

  return NewGrid, nx, ny, gt, proj


# def init_IMERG_config_pysteps():
#     """Function to initialize Pysteps for IMERG data
#         This has been adapted from Pysteps' tutorial colab notebook
    
#     """
#     # If the configuration file is placed in one of the default locations
#     # (https://pysteps.readthedocs.io/en/latest/user_guide/set_pystepsrc.html#configuration-file-lookup)
#     # it will be loaded automatically when pysteps is imported.
#     config_file_path = create_default_pystepsrc("pysteps_data")



#     # Import pysteps and load the new configuration file
#     import pysteps

#     _ = pysteps.load_config_file(config_file_path, verbose=True)
#     # The default parameters are stored in pysteps.rcparams.

#     # print(pysteps.rcparams.data_sources.keys())
#     pysteps.rcparams.data_sources['imerg'] = {'fn_ext': 'nc4',
#                                             'fn_pattern': 'PrecipRate_00.00_%Y%m%d-%H%M%S',
#                                             'importer': 'netcdf_pysteps',
#                                             'importer_kwargs': {},
#                                             'path_fmt': '%Y/%m/%d',
#                                             'root_path': '/content/IMERG/Flood_Ghana_032023/',
#                                             'timestep': 30}
#     print(pysteps.rcparams.data_sources['imerg'])