import glob

import pandas as pd
from utils import utilsRaster
import matplotlib as plt
import numpy as np

# # Import Montana SNOTEL Stations Information (Site Number, Longitude, and Latitude) as Data Frame (dfSNOTEL) from
# #   json dictionary
# dfSNOTEL = pd.read_json("/home/zachary/workspace/hydro_model/data/swecoords.json")
#
# # coords: Zip together Longitude (row 0) with Latitude (row 1) into dictionary of long, lat tuples.
# coords = zip(dfSNOTEL.values[0], dfSNOTEL.values[1])
#
# # Transforming long/lat into column and row of pixels in swe_*.tif files.
# # Squeezing the third dimension out of the raster, into an array
# # Grabbing swe values at each pixel for each raster
# datarow = []
# colNames = []
# dates = []
# for file in glob.glob("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2007-2017/swe_*.tif"):
#
#     swe = utilsRaster.RasterParameterIO(file)
#     colrow = [~swe.transform * c for c in coords]
#     swearray = swe.array.squeeze()
#     # values = [swearray[int(c[1]), int(c[0])] for c in colrow]
#     values = []
#
#     for c in range(len(colrow)):
#
#         try:
#             values.append(swearray[int(colrow[c][1]), int(colrow[c][0])])
#             date = rast.split('_')[-1].split('.')[0]
#
#             # Appending values to include date
#             datarow.append(values)
#             dates.append(date)
#
#             colNames.append(dfSNOTEL.columns[c])
#
# # Splitting date from swe_*.tif file names
#     date = file.split('_')[-1].split('.')[0]
#
# # Appending values to include date
#     datarow.append(values)
#     dates.append(date)
#
# # Dataframe with SWE values indexed and sorted by date, and added SNOTEL Station ID
# dfFinal = pd.DataFrame(datarow, index=pd.to_datetime(dates))
# dfFinal.columns = dfSNOTEL.columns
# dfFinal = dfFinal.sort_index()
#
# # print(dfFinal)
#
# # Plotting time-series for all SNOTEL Stations
# # dfFinal.plot()


dfSNOTEL = pd.read_json("/home/zachary/workspace/hydro_model/data/swecoords.json")

coords = zip(dfSNOTEL.values[0], dfSNOTEL.values[1])

datarow = []
colNames = []
dates = []

for rast in glob.glob("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2007-2017/swe_*.tif"):

    swe = utilsRaster.RasterParameterIO(rast)
    colrow = [~swe.transform * c for c in coords]
    swearray = swe.array.squeeze()

    # values = [swearray[int(c[1]), int(c[0])] for c in colrow]
    values = []

    for c in range(len(colrow)):

        try:
            values.append(swearray[int(colrow[c][1]), int(colrow[c][0])])
            date = rast.split('_')[-1].split('.')[0]

            # Appending values to include date
            datarow.append(values)
            dates.append(date)

            colNames.append(dfSNOTEL.columns[c])

        except IndexError:
            continue

# Dataframe with SWE values indexed and sorted by date, and added SNOTEL Station ID

dfFinal = pd.DataFrame(datarow, index=pd.to_datetime(dates))
dfFinal.columns = set(colNames)
dfFinal = dfFinal.sort_index()

model_swe = dfFinal
model_swe = model_swe[~model_swe.index.duplicated(keep='first')]
model_swe.columns = model_swe.columns.astype(int)
model_swe.sort_index(axis=1, inplace=True)
model_swe.index = pd.to_datetime(model_swe.index)

# Saving dataframe as csv
model_swe.to_csv('/home/zachary/workspace/hydro_model/data/model_swe_bitterroot_2007-2017.csv')