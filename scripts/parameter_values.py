import glob

import pandas as pd
from utils import utilsRaster, utilsVector
import matplotlib as plt
import numpy as np

# Changing Raster Values 2.31388863e+00  9.91941801e-01  7.26306045e+02  4.40534210e+00
#    3.94358970e-01  2.60426142e+01  4.49671518e+00  3.19718066e+02
#    4.58662178e+01  1.69716835e+01

# Changing value of Degree Day Factor (ddf) raster.
ddf = utilsRaster.RasterParameterIO("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/ddf_default.tif")
ddf.array = np.ones_like(ddf.array) *    1.76697626e+00
ddf.write_array_to_geotiff('/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/ddf.tif', ddf.array.squeeze())

# Changing value of Temperature Threshold (pp_temp_thres) raster.
pp_temp_thres = utilsRaster.RasterParameterIO("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/pp_temp_thres_default.tif")
pp_temp_thres.array = np.ones_like(pp_temp_thres.array) *     4.52887421e+00
pp_temp_thres.write_array_to_geotiff('/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/pp_temp_thres.tif', pp_temp_thres.array.squeeze())

# Changing value of Soil Max Water (soil_max_wat) raster.
soil_max_wat = utilsRaster.RasterParameterIO("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/soil_max_wat_default.tif")
soil_max_wat.array = np.ones_like(soil_max_wat.array) * 4.55264696e+02
soil_max_wat.write_array_to_geotiff("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/soil_max_wat.tif", soil_max_wat.array.squeeze())

# Changing value of Soil Beta (soil_beta) raster.
soil_beta = utilsRaster.RasterParameterIO("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/soil_beta_default.tif")
soil_beta.array = np.ones_like(soil_beta.array) * 3.66296017e-01
soil_beta.write_array_to_geotiff("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/soil_beta.tif", soil_beta.array.squeeze())

# Changing value of Limit for potential evapotranspiration (aet_lp_param) raster.
aet_lp_param = utilsRaster.RasterParameterIO("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/aet_lp_param_default.tif")
aet_lp_param.array = np.ones_like(aet_lp_param.array) * 5.43897401e-01
aet_lp_param.write_array_to_geotiff("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/aet_lp_param.tif", aet_lp_param.array.squeeze())
#
#
# # Changing Vector Values
#
# # Changing value of HBV Hydrologic Model parameters.
hbv = utilsVector.VectorParameterIO("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/subsout_default.shp")
#
lstDict = []
for feat in hbv.read_features():
#
#     # if feat['properties']['OBJECTID'] == 241:
#     #     feat['properties']['hbv_ck0'] = 22.8221
#     #     feat['properties']['hbv_ck1'] = 14.5353
#     #     feat['properties']['hbv_ck2'] = 550.469
#     #     feat['properties']['hbv_perc'] = 1.20006
#     #     feat['properties']['hbv_hl1'] = 50.
#     #     feat['properties']['hbv_pbase'] = 5.0000000E+00
#     #
#     # if feat['properties']['OBJECTID'] == 242:
#     #     feat['properties']['hbv_ck0'] = 22.8328
#     #     feat['properties']['hbv_ck1'] = 1.00000
#     #     feat['properties']['hbv_ck2'] = 307.605
#     #     feat['properties']['hbv_perc'] = 2.86341
#     #     feat['properties']['hbv_hl1'] = 5.0000000E+01
#     #     feat['properties']['hbv_pbase'] = 5.0000000E+00
#     #
#     # if feat['properties']['OBJECTID'] == 244:
#     #     feat['properties']['hbv_ck0'] = 14.0000
#     #     feat['properties']['hbv_ck1'] = 1.31065
#     #     feat['properties']['hbv_ck2'] = 550.567
#     #     feat['properties']['hbv_perc'] = 2.50424
#     #     feat['properties']['hbv_hl1'] = 50.
#     #     feat['properties']['hbv_pbase'] = 5.0000000E+00
#     #
#     # if feat['properties']['OBJECTID'] == 248:
    feat['properties']['hbv_ck0'] = 9.51740575e+00
    feat['properties']['hbv_ck1'] = 4.62809844e+01
    feat['properties']['hbv_ck2'] = 2.28356437e+02
    feat['properties']['hbv_perc'] = 5.26222360e+00
    feat['properties']['hbv_hl1'] = 6.11747402e+01
    feat['properties']['hbv_pbase'] = 5.0000000E+00

    # feat['properties']['hbv_hl1'] = 50.
    # feat['properties']['hbv_pbase'] = 5
#
#     if feat['properties']['OBJECTID'] == 241:
#         feat['properties']['hbv_ck0'] = 22.8328
#         feat['properties']['hbv_ck1'] = 14.5363
#         feat['properties']['hbv_ck2'] = 1000.
#         feat['properties']['hbv_perc'] = 0.500000
#
#     if feat['properties']['OBJECTID'] == 242:
#         feat['properties']['hbv_ck0'] = 22.8328
#         feat['properties']['hbv_ck1'] = 14.5363
#         feat['properties']['hbv_ck2'] = 1000.
#         feat['properties']['hbv_perc'] = 0.500000
#
#     if feat['properties']['OBJECTID'] == 244:
#         feat['properties']['hbv_ck0'] = 14.
#         feat['properties']['hbv_ck1'] = 1.31052
#         feat['properties']['hbv_ck2'] = 200.
#         feat['properties']['hbv_perc'] = 17.3000
#
#     if feat['properties']['OBJECTID'] == 248:
#         feat['properties']['hbv_ck0'] = 14.
#         feat['properties']['hbv_ck1'] = 1.31052
#         feat['properties']['hbv_ck2'] = 200.
#         feat['properties']['hbv_perc'] = 17.3000
#
    lstDict.append(feat)
#
#
hbv.write_dataset('/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/subsout.shp', params=lstDict)
#
#
# # Changing values of Muskingum-Cunge parameters

mkc = utilsVector.VectorParameterIO("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/rivout_default.shp")

lstDict = []
for feat in mkc.read_features():

    # if feat['properties']['OBJECTID'] == 241:
    #     feat['properties']['e'] = 3.5000000E-01
    #     feat['properties']['ks'] = 8.6400000E+04
    #
    # if feat['properties']['OBJECTID'] == 242:
    #     feat['properties']['e'] = 3.5000000E-01
    #     feat['properties']['ks'] = 8.64E+04
    #
    # if feat['properties']['OBJECTID'] == 244:
    #     feat['properties']['e'] = 3.5000000E-01
    #     feat['properties']['ks'] = 8.6400000E+04
    #
    # if feat['properties']['OBJECTID'] == 248:
    #     feat['properties']['e'] = 3.5000000E-01
    #     feat['properties']['ks'] = 8.6400000E+04

    feat['properties']['e'] = 0.5
    feat['properties']['ks'] = 172800.

    lstDict.append(feat)

mkc.write_dataset('/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/rivout.shp', params=lstDict)
