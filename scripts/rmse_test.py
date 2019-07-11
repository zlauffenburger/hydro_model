import pandas as pd
import json
from spotpy.objectivefunctions import nashsutcliffe as nse

# obs = pd.read_csv('/home/zachary/workspace/hydro_model/data/streamflow/2007-2017/pd_streamflow_2007-2017.csv').set_index('dateTime')
# sim = pd.read_json('/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2007-2017/streamflows.json').set_index("dateTime")
#
#
# # obs = obs.drop(['2008-02-29'])
#
# obs = obs.reset_index(drop=True)
# sim = sim.reset_index(drop=True)


class RMSE(object):
    def __init__(self, obs_path, sim_path):

        # with open(sim_path) as f:
        #     flows = json.load(f)
        self.real_run = pd.read_csv(obs_path).set_index('dateTime')
        self.real_run.index = pd.to_datetime(self.real_run.index)
        self.real_run.columns = self.real_run.columns.astype(int)
        # df = pd.DataFrame()
        # nodes = flows['nodes']
        # for node in nodes:
        #     id = node['id']
        #     if id in self.real_run.columns:
        #         date_list = node['dates']
        #         flows = pd.DataFrame(date_list, columns=['date', 'flow'])
        #         flows = flows.rename(columns={'flow': id})
        #         flows = flows.set_index(['date'])
        #         flows.index = pd.to_datetime(flows.index)
        #
        #         df = pd.concat([df, flows], axis=1)
        # self.model_run = df

        self.model_run = pd.read_csv(sim_path).set_index('dateTime')
        self.model_run.index = pd.to_datetime(self.model_run.index)
        self.model_run.columns = self.model_run.columns.astype(int)

        # only matching dates are compared
        bools = [i in self.model_run.index.values for i in self.real_run.index.values]
        self.model_run = self.model_run[bools]

    def get_ids(self):
        """Returns IDs of gages"""
        return self.model_run.columns.values

    def get_sim_data(self):
        """Returns pandas dataframe of modeled data"""
        return self.model_run

    def get_obs_data(self):
        """Returns pandas dataframe of observation data"""
        return self.real_run


    def root_mean_square_error(self, obs, sim):
        obs = self.real_run.reset_index(drop=True)
        sim = self.model_run.reset_index(drop=True)

        return ((obs - sim) ** 2).mean() ** 0.5


# Main code
obs_path = '/home/zachary/workspace/lineartransfer/data/pd_streamflow_lt_2016-2017.csv'
sim_path = '/home/zachary/workspace/lineartransfer/error/files_concat/model_runoff_run56.csv'

rmse = RMSE(obs_path, sim_path)
ids = rmse.get_ids()
obs_data = rmse.get_obs_data()
sim_data = rmse.get_sim_data()
rmse2 = rmse.root_mean_square_error(obs_data, sim_data)

print rmse2[88]
print rmse2[100]


