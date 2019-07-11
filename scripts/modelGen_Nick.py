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
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import nashsutcliffe as nse
import hydrovehicle
import signal, time

# np.random.seed(123)  # hangs on first run

class TimeoutError (RuntimeError):
    pass

class DawuapMonteCarlo(object):

    def __init__(self, model_directory, user_inputs=None, rand_size=1):

        self.user_inputs = user_inputs
        self.model_directory = model_directory
        self.rand_size = rand_size
        self.run_number = 0
        self.use_dir = None
        self.err_dir = None
        self.rand_array = None
        self.model_swe = None
        self.real_swe = None
        self.model_run = None
        self.real_run = None


    def _get_raster_values(self):

        out_df = pd.DataFrame()

        with open(os.path.join(self.model_directory, 'params', 'param_files_test.json')) as json_data:
            dat = json.load(json_data)

        for feat in dat.values():

            rast_name = os.path.join(self.model_directory, 'params', feat)

            arr = utilsRaster.RasterParameterIO(rast_name).array
            val = np.unique(arr.squeeze())
            by_val = val/2.

            out_values = np.random.uniform(float(val) - float(by_val), float(val) + float(by_val), self.rand_size)
            out_values = pd.DataFrame({feat: out_values})

            out_df = pd.concat([out_df, out_values], axis=1)

        return out_df

    def _get_basin_values(self):

        out_df = pd.DataFrame()

        dat = utilsVector.VectorParameterIO(os.path.join(self.model_directory, 'params/subsout.shp'))
        hbv_ck0 = np.zeros(1)
        hbv_ck1 = np.zeros(1)
        hbv_ck2 = np.zeros(1)
        hbv_hl1 = np.zeros(1)
        hbv_perc = np.zeros(1)
        hbv_pbase = np.zeros(1)

        for feat in dat.read_features():
            hbv_ck0 = np.append(hbv_ck0, feat['properties']['hbv_ck0'])
            hbv_ck1 = np.append(hbv_ck1, feat['properties']['hbv_ck1'])
            hbv_ck2 = np.append(hbv_ck2, feat['properties']['hbv_ck2'])
            hbv_hl1 = np.append(hbv_hl1, feat['properties']['hbv_hl1'])
            hbv_perc = np.append(hbv_perc, feat['properties']['hbv_perc'])
            hbv_pbase = np.append(hbv_pbase, feat['properties']['hbv_pbase'])

        hbv_ck0 = np.unique(hbv_ck0[hbv_ck0 != 0.])
        hbv_ck1 = np.unique(hbv_ck1[hbv_ck1 != 0.])
        hbv_ck2 = np.unique(hbv_ck2[hbv_ck2 != 0.])
        hbv_hl1 = np.unique(hbv_hl1[hbv_hl1 != 0.])
        hbv_perc = np.unique(hbv_perc[hbv_perc != 0.])
        hbv_pbase = np.unique(hbv_pbase[hbv_pbase != 0.])

        dat = {"hbv_ck0": hbv_ck0,
               "hbv_ck1": hbv_ck1,
               "hbv_ck2": hbv_ck2,
               "hbv_hl1": hbv_hl1,
               "hbv_perc": hbv_perc}

        for val in dat:

            by_val = dat[val] / 2.

            out_values = np.random.uniform(float(dat[val]) - float(by_val), float(dat[val]) + float(by_val), self.rand_size)
            out_values = pd.DataFrame({val: out_values})

            out_df = pd.concat([out_df, out_values], axis=1)

        by_val = hbv_pbase / 2.
        by_val = math.ceil(by_val)

        out_values = np.random.randint(int(hbv_pbase) - int(by_val), int(hbv_pbase) + int(by_val), self.rand_size)
        out_values = pd.DataFrame({'hbv_pbase': out_values})

        return pd.concat([out_df, out_values], axis=1)

    def _get_river_values(self):

        # These values might change. Keep in mind for later

        e = np.random.uniform(0, 0.5, self.rand_size)
        ks = np.random.uniform(41200., 123600., self.rand_size)

        return pd.DataFrame({'e': e,
                             'ks': ks})

    def _use_user_params(self):

        out_dict = {}
        for feat in self.user_inputs:

            if feat == 'hbv_pbase':

                out_vals = np.random.randint(self.user_inputs[feat][0], self.user_inputs[feat][1], self.rand_size)
                out_dict[feat] = out_vals

            else:
                out_vals = np.random.uniform(self.user_inputs[feat][0], self.user_inputs[feat][1], self.rand_size)
                out_dict[feat] = out_vals

        return pd.DataFrame(out_dict)

    def _gen_error_dir(self):

        default_number = 1
        default_name = os.path.join(self.model_directory, 'error', 'error' + str(default_number))

        while os.path.isdir(default_name):
            default_number += 1
            default_name = os.path.join(self.model_directory, 'error', 'error' + str(default_number))

        os.makedirs(default_name)
        self.err_dir = default_name

        blank_error = pd.DataFrame()

        blank_error.to_csv(os.path.join(self.err_dir, 'swe_error.csv'))
        blank_error.to_csv(os.path.join(self.err_dir, 'run_error.csv'))

    def _gen_use_dir(self):

        default_number = 1
        default_name = os.path.join(self.model_directory, 'temp', 'temp' + str(default_number))

        while os.path.isdir(default_name):
            default_number += 1
            default_name = os.path.join(self.model_directory, 'temp', 'temp' + str(default_number))

        os.makedirs(default_name)
        self.use_dir = default_name
        os.chdir(self.use_dir)

    def _set_rand_array(self):

        if self.user_inputs is None:

            self.rand_array = pd.concat([self._get_river_values(),
                                         self._get_basin_values(),
                                         self._get_raster_values()], axis=1)

            self.rand_array.to_csv(os.path.join(self.err_dir, 'rand_array.csv'))

        else:
            self.rand_array = self._use_user_params()
            self.rand_array.to_csv(os.path.join(self.err_dir, 'rand_array.csv'))

    def _clean_up(self):

        rmtree(self.use_dir)
        self.use_dir = None
        os.chdir(self.model_directory)

    def _create_raster_parameters(self):

        with open(os.path.join(self.model_directory, 'params', 'param_files_test.json')) as json_data:
            dat = json.load(json_data)

        for feat in dat:

            value = self.rand_array[feat].iloc[self.run_number]
            rast = utilsRaster.RasterParameterIO(os.path.join(self.model_directory, 'params', dat[feat]))
            rast.array.fill(value)

            rast.write_array_to_geotiff(dat[feat], np.squeeze(rast.array))

        copyfile(os.path.join(self.model_directory, 'params/param_files_test.json'),
                 os.path.join(os.getcwd(), 'rast_params.json'))

    def _create_basin_parameters(self):

        sub_vec = utilsVector.VectorParameterIO(os.path.join(self.model_directory, 'params/subsout.shp'))

        lstDict = []
        for feat in sub_vec.read_features():
            feat['properties']['hbv_ck0'] = self.rand_array['hbv_ck0'].iloc[self.run_number]
            feat['properties']['hbv_ck1'] = self.rand_array['hbv_ck1'].iloc[self.run_number]
            feat['properties']['hbv_ck2'] = self.rand_array['hbv_ck2'].iloc[self.run_number]
            feat['properties']['hbv_hl1'] = self.rand_array['hbv_hl1'].iloc[self.run_number]
            feat['properties']['hbv_perc'] = self.rand_array['hbv_perc'].iloc[self.run_number]
            feat['properties']['hbv_pbase'] = self.rand_array['hbv_pbase'].iloc[self.run_number]

            lstDict.append(feat)

        sub_vec.write_dataset('basin_out.shp', params=lstDict)

    def _create_river_parameters(self):

        riv_vec = utilsVector.VectorParameterIO(os.path.join(self.model_directory, 'params/rivout.shp'))

        lstDict = []
        for feat in riv_vec.read_features():

            feat['properties']['e'] = self.rand_array['e'].iloc[self.run_number]
            feat['properties']['ks'] = self.rand_array['ks'].iloc[self.run_number]

            lstDict.append(feat)

        riv_vec.write_dataset('river_out.shp', params=lstDict)

    def _write_random_parameters(self):

        self._gen_use_dir()
        self._create_raster_parameters()
        self._create_basin_parameters()
        self._create_river_parameters()

    def _force_symlink(self, file1, file2):
        try:
            os.symlink(file1, file2)
        except OSError, e:
            if e.errno == errno.EEXIST:
                os.remove(file2)
                os.symlink(file1, file2)

    def _run_singular_model(self, spinup=2):
        """
        :params spinup: number of times (int) to spin the model up
        """

        args = Namespace(init_date='09/01/2016',
                        precip=os.path.join(self.model_directory, 'params/precip_F2016-09-01_T2017-08-31.nc'),
                        tmin=os.path.join(self.model_directory, 'params/tempmin_F2016-09-01_T2017-08-31.nc'),
                        tmax=os.path.join(self.model_directory, 'params/tempmax_F2016-09-01_T2017-08-31.nc'),
                        params='rast_params.json',
                        network_file='river_out.shp',
                        basin_shp='basin_out.shp',
                        restart=False,
                        econengine=None)
        if spinup > 0:
            hydrovehicle.main(args)
            print 'Done with initialization run'
            args.restart = True
            k = 1
            while k < spinup:
                restart_files = glob.glob(self.use_dir + "/*.pickled")
                for r in restart_files:
                    rname = r.split("/")[-1]
                    self._force_symlink(r, self.model_directory + rname)
                hydrovehicle.main(args)
                print 'Done with spin #' + str(k)
                k += 1
        else:
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
        real_swe = pd.read_csv(os.path.join(self.model_directory, 'data/snotel_swe_2016-2017.csv'),
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

        self.real_run = pd.read_csv(os.path.join(self.model_directory, 'data/streamflow/2016-2017/pd_streamflow_2016-2017.csv'))
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


    # def rmse(self, obs, mod):
    #     """Calculates root mean squared error between two numpy arrays."""
    #     e = obs - mod
    #     se = e**2
    #     mse = se.mean()
    #     return np.sqrt(mse)

    def _generate_error_statistics(self):

        self._format_swe_data()
        self._format_sf_data()

        # TEMPORARY FIX
        self.real_run = self.real_run[self.model_run.columns.tolist()]
        self.real_run = self.real_run.dropna(thresh=2)

        bools = [i in self.real_run.index.values for i in self.model_run.index.values]
        self.model_run = self.model_run[bools]

        # END FIX

        run_error = pd.read_csv(os.path.join(self.err_dir, 'run_error.csv'), index_col=0)
        run_error.columns = run_error.columns.astype(str)
        swe_error = pd.read_csv(os.path.join(self.err_dir, 'swe_error.csv'), index_col=0)
        swe_error.columns = swe_error.columns.astype(str)

        run_dict = {'run_number': self.run_number}
        swe_dict = {'run_number': self.run_number}

        for col in self.model_run:

            run_dict[str(col)] = nse(self.real_run[col].values, self.model_run[col].values)

        for col in self.model_swe:

            swe_dict[str(col)] = nse(self.real_swe[col].values, self.model_swe[col].values)

        run_error = run_error.append(pd.DataFrame(run_dict, index=[1]), ignore_index=True)
        run_error.columns = run_error.columns.astype(str)
        swe_error = swe_error.append(pd.DataFrame(swe_dict, index=[1]), ignore_index=True)
        swe_error.columns = swe_error.columns.astype(str)

        run_error.to_csv(os.path.join(self.err_dir, 'run_error.csv'))
        swe_error.to_csv(os.path.join(self.err_dir, 'swe_error.csv'))
        self.model_run.to_csv(os.path.join(self.err_dir, 'model_runoff.csv'))
        self.model_swe.to_csv(os.path.join(self.err_dir, 'model_swe.csv'))
        # self.real_swe.to_csv(os.path.join(self.err_dir, 'real_swe.csv'))
        # self.real_run.to_csv(os.path.join(self.err_dir, 'real_run.csv'))
    
    def handler (self, signum, frame):
        """This is used to handle timeout in run_model()"""
        raise TimeoutError()

    def run_model(self):

        signal.signal(signal.SIGALRM, self.handler)
        self._gen_error_dir()
        self._set_rand_array()

        i = 0
        while i < self.rand_size:
            try:
                self.run_number = i
                i += 1
                signal.alarm(3*20)
                self._write_random_parameters()
                self._run_singular_model()
                self._generate_error_statistics()
                self._clean_up()
                print 'You are on simulation number ' + str(i)
            except TimeoutError as ex:
                print 'Timeout Error'
                


# model_inputs = {'pp_temp_thres': [-1.5, 1.5],
#                 'ddf': [0.5, 5.],
#                 'soil_max_wat': [50., 300.],
#                 'soil_beta': [0.5, 4.],
#                 'aet_lp_param': [0.1, 0.95],
#                 'hbv_pbase': [3, 10],
#                 'hbv_hl1': [1., 30.],
#                 'hbv_ck0': [0.25, 5.],
#                 'hbv_ck1': [3., 40.],
#                 'hbv_ck2': [50., 500.],
#                 'e': [0.1, 0.4],
#                 'hbv_perc': [3., 40.],
#                 'ks': [41200., 123600.]}

model_inputs = {'pp_temp_thres': [1.2, 1.7],
                'ddf': [3.7, 4.0],
                'soil_max_wat': [170., 300.],
                'soil_beta': [0.5, 4.0],
                'aet_lp_param': [0.4, 0.6],
                'hbv_pbase': [4, 6],
                'hbv_hl1': [50., 70.],
                'hbv_ck0': [5., 15.5],
                'hbv_ck1': [3., 10.],
                'hbv_ck2': [300., 550.],
                'hbv_perc': [1., 35.],
                'e': [0.3, 0.4],
                'ks': [30000., 90000.]}

model = DawuapMonteCarlo("/home/zachary/workspace/hydro_model", model_inputs, 1000)
# print(model.model_directory)
# os.chdir("/home/geoscigrad/workspace/hydro_model/temp/temp1")
# model.err_dir = '/home/geoscigrad/workspace/hydro_model/error/error1'
# model.use_dir= '/home/geoscigrad/workspace/hydro_model/temp/temp1'
#
# model._generate_error_statistics()
model.run_model()




