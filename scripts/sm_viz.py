import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


sim_path = '/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2007-2017/upper_soil.json'

with open(sim_path) as f:
    sm = json.load(f)
df = pd.DataFrame()
nodes = sm['nodes']
for node in nodes:
    id = node['id']
    date_list = node['dates']
    sm = pd.DataFrame(date_list, columns=['date', 'flow'])
    sm = sm.rename(columns={'flow': id})
    sm = sm.set_index(['date'])
    sm.index = pd.to_datetime(sm.index)

    df = pd.concat([df, sm], axis=1)


#     def get_ids(self):
#         """Returns IDs of gages"""
#         return self.model_run.columns.values
#
#     def get_sim_data(self):
#         """Returns pandas dataframe of modeled data"""
#         return self.model_run
#
#     def get_obs_data(self):
#         """Returns pandas dataframe of observation data"""
#         return self.real_run
#
#     def plot_single_comparison(self, id):
#         """Plots a single timeseries for the specified gage id"""
#         obs_data = self.get_obs_data()
#         sim_data = self.get_sim_data()
#         obs_id = obs_data[id]
#         sim_id = sim_data[id]
#         merge_id = pd.DataFrame({'obs': obs_id, 'sim': sim_id})
#         merge_id.plot()
#         plt.title('Gage ID# ' + str(id))
#         plt.show()
#
#     def plot_all_comparisons(self):
#         """
#         Plots timeseries for (almost) all of the gages. In order to
#         keep the subplot array even, it drops the last 4 gages.
#         """
#         sim_data = self.get_sim_data()
#         obs_data = self.get_obs_data()
#         obs_data_2 = obs_data[sim_data.columns]
#
#         for col in sim_data.columns:
#             plt.plot(sim_data.index, sim_data[col], label='simulated')
#             plt.plot(obs_data_2.index, obs_data_2[col], label='observed')
#             plt.legend()
#             # plt.savefig("str_" + str(col)  + ".png")
#             plt.close()
#
#         fig, ax = plt.subplots(len(ids) / 10, 10)
#         k = 0
#         for i, ax_row in enumerate(ax):
#             for j, axes in enumerate(ax_row):
#                 id = ids[k]
#                 axes.set_title('Gage ID# ' + str(id))
#                 axes.title.set_size(5)
#                 x_axis = axes.get_xaxis()
#                 x_axis.set_visible(False)
#                 obs_id = obs_data[id]
#                 sim_id = sim_data[id]
#                 merge_id = pd.DataFrame({'obs': obs_id, 'sim': sim_id})
#                 merge_id.plot(ax=axes, legend=False, fontsize=4)
#                 k += 1
#         plt.legend(bbox_to_anchor=(-9, 10.5), ncol=2)
#         plt.show()
#
#
# # Main code
# obs_path = '/home/zachary/workspace/hydro_model/data/streamflow/2016-2017/pd_streamflow_2016-2017.csv'

# ids = sv.get_ids()
# # sv.plot_single_comparison(ids[1])
# # sv.plot_all_comparisons()
# sv.plot_single_comparison(241)
# sv.plot_single_comparison(244)

df.to_csv('/home/zachary/Maps_Figures/sm_2007-2017.csv')
