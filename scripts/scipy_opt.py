import os
import errno
import json
import pandas as pd
from utils import utilsVector
from utils import utilsRaster
import numpy as np
import math
from shutil import copyfile, rmtree
from argparse import Namespace
import glob
from spotpy.objectivefunctions import mse
import hydrovehicle
import signal, time
import scipy.optimize as opt


# Retrieve Observations
real_Q = pd.read_csv('/home/zachary/workspace/hydro_model/data/streamflow/2016-2017/pd_streamflow_2016-2017.csv').set_index('dateTime')

class DawuapMinimize(object):
    def __init__(self, model_directory):

        self.model_directory = model_directory
        self.use_dir = None
        self.err_dir = None
        self.model_swe = None
        self.real_swe = None
        self.model_run = None
        self.real_run = None
        self.run_number = 0



    def _gen_error_dir(self):
        default_number = 1
        default_name = os.path.join(self.model_directory, 'errorOpt', 'errorOpt' + str(default_number))

        while os.path.isdir(default_name):
            default_number += 1
            default_name = os.path.join(self.model_directory, 'errorOpt', 'errorOpt' + str(default_number))

        os.makedirs(default_name)
        self.err_dir = default_name

        blank_error = pd.DataFrame()

        blank_error.to_csv(os.path.join(self.err_dir, 'swe_errorOpt.csv'))
        blank_error.to_csv(os.path.join(self.err_dir, 'run_errorOpt.csv'))

    def _gen_use_dir(self):

        default_number = 1
        default_name = os.path.join(self.model_directory, 'tempOpt', 'tempOpt' + str(default_number))

        while os.path.isdir(default_name):
            default_number += 1
            default_name = os.path.join(self.model_directory, 'tempOpt', 'tempOpt' + str(default_number))

        os.makedirs(default_name)
        self.use_dir = default_name
        os.chdir(self.use_dir)

    def _run_singular_model(self):
        args = Namespace(init_date='09/01/2007',
                     precip=os.path.join(self.model_directory, '/home/zachary/workspace/hydro_model/optimize/precip_F2016-09-01_T2017-08-31.nc'),
                     tmin=os.path.join(self.model_directory, '/home/zachary/workspace/hydro_model/optimize/tempmin_F2016-09-01_T2017-08-31.nc'),
                     tmax=os.path.join(self.model_directory, '/home/zachary/workspace/hydro_model/optimize/tempmax_F2016-09-01_T2017-08-31.nc'),
                     params='/home/zachary/workspace/hydro_model/optimize/param_files_test.json',
                     network_file='/home/zachary/workspace/hydro_model/optimize/rivout.shp',
                     basin_shp='/home/zachary/workspace/hydro_model/optimize/subsout.shp',
                     restart=False,
                     econengine=None)
        hydrovehicle.main(args)


    def _get_model_swe(self):
        dfSNOTEL = pd.read_json(os.path.join(self.model_directory, 'data/swecoords.json'))

        coords = zip(dfSNOTEL.values[0], dfSNOTEL.values[1])

        datarow = []
        colNames = []
        dates = []

        for rast in glob.glob("./swe_*.tif"):

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

            self.model_swe = dfFinal


    def _format_swe_data(self):
        self._get_model_swe()
        self.model_swe = self.model_swe[~self.model_swe.index.duplicated(keep='first')]
        real_swe = pd.read_csv(os.path.join(self.model_directory, 'data/snotel_swe_2007-2017.csv'),
                            index_col='date')
        real_swe.columns = real_swe.columns.astype(int)
        real_swe.sort_index(axis=1, inplace=True)
        real_swe.index = pd.to_datetime(real_swe.index)

        self.model_swe.columns = self.model_swe.columns.astype(int)
        self.model_swe.sort_index(axis=1, inplace=True)
        self.model_swe.index = pd.to_datetime(self.model_swe.index)

        bools = [i in self.model_swe.index.values for i in real_swe.index.values]
        real_swe = real_swe[bools]

        select_columns = self.model_swe.columns.values.tolist()

        self.real_swe = real_swe[select_columns]


    def _format_sf_data(self):

        self.real_run = pd.read_csv(os.path.join(self.model_directory, 'data/streamflow/2007-2017/pd_streamflow_2007-2017.csv'))
        self.real_run = self.real_run.set_index('dateTime')
        self.real_run.index = pd.to_datetime(self.real_run.index)
        self.real_run.columns = self.real_run.columns.astype(int)

        with open('./streamflows.json') as f:
            flows = json.load(f)

        df = pd.DataFrame()
        nodes = flows['nodes']

        for node in nodes:

            id = node['id']
            if id in self.real_run.columns:
                date_list = node['dates']

                flows = pd.DataFrame(date_list, columns=['date', 'flow'])
                flows = flows.rename(columns={'flow': id})
                flows = flows.set_index(['date'])
                flows.index = pd.to_datetime(flows.index)

                df = pd.concat([df, flows], axis=1)

            self.model_run = df

    # ensures that only matching dates are compared
            bools = [i in self.model_run.index.values for i in self.real_run.index.values]
            self.model_run = self.model_run[bools]

    def _generate_error_statistics(self):
        self._format_swe_data()
        self._format_sf_data()

    # TEMPORARY FIX
        self.real_run = self.real_run[self.model_run.columns.tolist()]
        self.real_run = self.real_run.dropna(thresh=2)

        bools = [i in self.real_run.index.values for i in self.model_run.index.values]
        self.model_run = self.model_run[bools]

    # END FIX

        run_error = pd.read_csv(os.path.join(self.err_dir, 'run_errorOpt.csv'), index_col=0)
        run_error.columns = run_error.columns.astype(str)
        swe_error = pd.read_csv(os.path.join(self.err_dir, 'swe_errorOpt.csv'), index_col=0)
        swe_error.columns = swe_error.columns.astype(str)

        run_dict = {'run_number': self.run_number}
        swe_dict = {'run_number': self.run_number}

        for col in self.model_run:
            run_dict[str(col)] = mse(self.real_run[col].values, self.model_run[col].values)

        for col in self.model_swe:
            swe_dict[str(col)] = mse(self.real_swe[col].values, self.model_swe[col].values)

        run_error = run_error.append(pd.DataFrame(run_dict, index=[1]), ignore_index=True)
        run_error.columns = run_error.columns.astype(str)
        swe_error = swe_error.append(pd.DataFrame(swe_dict, index=[1]), ignore_index=True)
        swe_error.columns = swe_error.columns.astype(str)

        run_error.to_csv(os.path.join(self.err_dir, 'run_errorOpt.csv'))
        swe_error.to_csv(os.path.join(self.err_dir, 'swe_errorOpt.csv'))


# Run Hydromodel
    def run_model(self):
        self._gen_error_dir()
        self._run_singular_model()
        self._generate_error_statistics()


# Set initial parameter guess
pp_temp_thres = 2.
ddf = 0.02
soil_max_wat = 50.
soil_beta = 0.5
aet_lp_param = 0.5
hbv_pbase = 5.
hbv_hl1 = 50.
hbv_ck0 = 10.
hbv_ck1 = 50.
hbv_ck2 = 10000.
hbv_perc = 50.
e = 0.35
ks = 82400.

initial_parameter_guess = (pp_temp_thres, ddf, soil_max_wat, soil_beta, aet_lp_param, hbv_pbase, hbv_hl1, hbv_ck0, hbv_ck1, hbv_ck2, hbv_perc, e, ks)

# set model intital condition and number of time steps
q0 = real_Q.values
num_time_steps = real_Q.size




# Main code
model = DawuapMinimize('/home/zachary/workspace/hydro_model/')
model.run_model()

#run minimize routine from scipy optimize to find parameters that minimize sse

sol = opt.minimize(sse, np.ndarray(pp_temp_thres, ddf, soil_max_wat, soil_beta, aet_lp_param, hbv_pbase, hbv_hl1, hbv_ck0, hbv_ck1, hbv_ck2, hbv_perc, e, ks),
                   args=(q0, num_time_steps),
                   method='L-BFGS-B'
                   )

print sol