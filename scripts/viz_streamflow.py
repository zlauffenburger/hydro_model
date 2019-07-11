import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

class StreamViz(object):
    def __init__(self, obs_path, sim_path):
        
        with open(sim_path) as f:
            flows = json.load(f)
        self.real_run = pd.read_csv(obs_path).set_index('dateTime')
        self.real_run.index = pd.to_datetime(self.real_run.index)
        self.real_run.columns = self.real_run.columns.astype(int)
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

    def plot_single_comparison(self, id):
        """Plots a single timeseries for the specified gage id"""
        obs_data = self.get_obs_data()
        sim_data = self.get_sim_data()
        obs_id = obs_data[id]
        sim_id = sim_data[id]
        merge_id = pd.DataFrame({'obs': obs_id, 'sim': sim_id})
        merge_id.plot(fontsize=40)
        plt.title('Gage ID# ' + str(id), fontsize=50)
        plt.legend(prop={'size': 50})
        plt.ylabel('Discharge (m3/sec)', fontsize=40)
        plt.rcParams.update({'xtick.labelsize': 'x-large'})
        plt.rcParams.update({'ytick.labelsize': 'x-large'})
        plt.show()

    def plot_all_comparisons(self):
        """
        Plots timeseries for (almost) all of the gages. In order to 
        keep the subplot array even, it drops the last 4 gages. 
        """
        sim_data = self.get_sim_data()
        obs_data = self.get_obs_data()
        obs_data_2 = obs_data[sim_data.columns]

        for col in sim_data.columns:
            plt.plot(sim_data.index, sim_data[col], label='simulated')
            plt.plot(obs_data_2.index, obs_data_2[col], label='observed')
            plt.legend()
            #plt.savefig("str_" + str(col)  + ".png")
            plt.close()


        fig, ax = plt.subplots(len(ids)/10, 10)
        k = 0
        for i, ax_row in enumerate(ax):
            for j, axes in enumerate(ax_row):
                id = ids[k]
                axes.set_title('Gage ID# ' + str(id))
                axes.title.set_size(5)
                x_axis = axes.get_xaxis()
                x_axis.set_visible(False)
                obs_id = obs_data[id]
                sim_id = sim_data[id]
                merge_id = pd.DataFrame({'obs': obs_id, 'sim': sim_id})
                merge_id.plot(ax=axes, legend=False, fontsize=4)
                k += 1
        plt.legend(bbox_to_anchor = (-9, 10.5), ncol=2)
        plt.show()


# Main code
obs_path = '/home/zachary/workspace/lineartransfer/data/pd_streamflow_lt_2016-2017.csv'
sim_path = '/home/zachary/workspace/dawuaphydroengine/docs/example/streamflows.json'
sv = StreamViz(obs_path, sim_path)
ids = sv.get_ids()
#sv.plot_single_comparison(ids[1])
#sv.plot_all_comparisons()
sv.plot_single_comparison(242)
sv.plot_single_comparison(317)


# Calculate Error
# gage13obs = sv.real_run[13]
# gage13obs = gage13obs.drop([-1])
#
# print gage13obs
