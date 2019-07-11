import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


class SweViz(object):
    def __init__(self, obs_path, sim_path):


        self.real_swe = pd.read_csv(obs_path).set_index('date')
        self.real_swe.index = pd.to_datetime(self.real_swe.index)
        self.real_swe.columns = self.real_swe.columns.astype(int)

        self.model_swe = pd.read_csv(sim_path, index_col=0)
        self.model_swe.index = pd.to_datetime(self.model_swe.index)
        self.model_swe.columns = self.model_swe.columns.astype(int)

        # only matching dates are compared
        bools = [i in self.model_swe.index.values for i in self.real_swe.index.values]
        self.model_swe = self.model_swe[bools]

    def get_ids(self):
        """Returns IDs of snotel"""
        return self.model_swe.columns.values

    def get_sim_data(self):
        """Returns pandas dataframe of modeled data"""
        return self.model_swe

    def get_obs_data(self):
        """Returns pandas dataframe of observation data"""
        return self.real_swe

    def plot_single_comparison(self, id):
        """Plots a single timeseries for the specified snotel id"""
        obs_data = self.get_obs_data()
        sim_data = self.get_sim_data()
        obs_id = obs_data[id]
        sim_id = sim_data[id]
        merge_id = pd.DataFrame({'obs': obs_id, 'sim': sim_id})
        merge_id.plot()
        plt.title('SNOTEL ID# ' + str(id), fontsize=50)
        plt.legend(prop={'size': 50})
        plt.ylabel('Snow Water Equivalent (mm)', fontsize=40)
        plt.rcParams.update({'xtick.labelsize': 'x-large'})
        plt.rcParams.update({'ytick.labelsize': 'x-large'})
        plt.show()

    def plot_all_comparisons(self):
        """
        Plots timeseries for (almost) all of the snotel. In order to
        keep the subplot array even, it drops the last 4 gages.
        """
        sim_data = self.get_sim_data()
        obs_data = self.get_obs_data()
        obs_data_2 = obs_data[sim_data.columns]

        for col in sim_data.columns:
            plt.plot(sim_data.index, sim_data[col], label='simulated')
            plt.plot(obs_data_2.index, obs_data_2[col], label='observed')
            plt.legend()
            # plt.savefig("str_" + str(col)  + ".png")
            plt.close()

        fig, ax = plt.subplots(len(ids) / 10, 10)
        k = 0
        for i, ax_row in enumerate(ax):
            for j, axes in enumerate(ax_row):
                id = ids[k]
                axes.set_title('SNOTEL ID# ' + str(id))
                axes.title.set_size(5)
                x_axis = axes.get_xaxis()
                x_axis.set_visible(False)
                obs_id = obs_data[id]
                sim_id = sim_data[id]
                merge_id = pd.DataFrame({'obs': obs_id, 'sim': sim_id})
                merge_id.plot(ax=axes, legend=False, fontsize=4)
                k += 1
        plt.legend(bbox_to_anchor=(-9, 10.5), ncol=2)
        plt.show()


# Main code
obs_path = '/home/zachary/workspace/hydro_model/data/snotel_swe_2007-2017.csv'
sim_path = '/home/zachary/workspace/hydro_model/data/model_swe_bitterroot_2007-2017.csv'
sv = SweViz(obs_path, sim_path)
ids = sv.get_ids()
# sv.plot_single_comparison(ids[1])
# sv.plot_all_comparisons()
sv.plot_single_comparison(433)
sv.plot_single_comparison(662)
sv.plot_single_comparison(760)
sv.plot_single_comparison(835)
sv.plot_single_comparison(836)

# Calculate Error
# gage13obs = sv.real_run[13]
# gage13obs = gage13obs.drop([-1])
#
# print gage13obs
