import pandas
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns
import os
import subprocess
from collections import namedtuple
import logging
import glob

data_for_phy_path = r"S:\data_for_quality_rating"#r"C:\Users\svc_neuropix\Documents\data_for_phy2"
sorted_data_directory = r"D:\test_phy_416356"

probe_list = ['C']#['A', 'B', 'C', 'D', 'E', 'F']
experiment_directories = []
experiment_directories.append(r"\\10.128.50.60\SD4")
experiment_directories.append(r"\\10.128.50.60\SD4.2")
experiment_directories.append(r"\\10.128.50.77\sd5")
experiment_directories.append(r"\\10.128.50.77\sd5.2")
experiment_directories.append(r"\\10.128.50.77\sd5.3")
experiment_directories.append(r'\\10.128.50.77\sd5.3\RE-SORT')

data_file_params = namedtuple('data_file_params',['relpath','upload','sorting_step'])

relpaths = {
                'lfp':r"continuous\Neuropix-3a-100.1",
                'spikes':r"continuous\Neuropix-3a-100.0",
                'events':r"events\Neuropix-3a-100.0\TTL_1",
                'empty':""
                    }
        
data_files = {
      "probe_info.json":data_file_params('empty',False,'depth_estimation'),
      "channel_states.npy":data_file_params('events',False,'extraction'),
      "event_timestamps.npy":data_file_params('events',False,'extraction'),
      r"continuous\Neuropix-3a-100.1\continuous.dat":data_file_params('empty',False,'extraction'),
      "lfp_timestamps.npy":data_file_params('lfp',False,'sorting'),
      "amplitudes.npy":data_file_params('spikes',True,'sorting'),
      "spike_times.npy":data_file_params('spikes',True,'sorting'),
          "mean_waveforms.npy":data_file_params('spikes',True,'mean waveforms'),
          "spike_clusters.npy":data_file_params('spikes',True,'sorting'),
          "spike_templates.npy":data_file_params('spikes',True,'sorting'),
          "templates.npy":data_file_params('spikes',True,'sorting'),
          "whitening_mat.npy":data_file_params('spikes',True,'sorting'),
          "whitening_mat_inv.npy":data_file_params('spikes',True,'sorting'),
          "templates_ind.npy":data_file_params('spikes',True,'sorting'),
          "similar_templates.npy":data_file_params('spikes',True,'sorting'),
          "metrics.csv":data_file_params('spikes',True,'metrics'),
          "new_metrics.csv":data_file_params('spikes',True,'metrics'),
          "channel_positions.npy":data_file_params('spikes',True,'sorting'),
          "cluster_group.tsv":data_file_params('spikes',True,'sorting'),
          "channel_map.npy":data_file_params('spikes',True,'sorting'),
          "params.py":data_file_params('spikes',True,'sorting'),
      "probe_depth.png":data_file_params("empty",False,'depth estimation'),
      r"continuous\Neuropix-3a-100.0\continuous.dat":data_file_params('empty',False,'extraction'),
      "residuals.dat":data_file_params('spikes',False,'median subtraction'),
      "pc_features.npy":data_file_params('spikes',True,'sorting'),
      "template_features.npy":data_file_params('spikes',True,'sorting'),
      "rez2.mat":data_file_params('spikes',False,'sorting'),
      "rez.mat":data_file_params('spikes',False,'sorting'),
      "pc_feature_ind.npy":data_file_params('spikes',True,'sorting'),
      "template_feature_ind.npy":data_file_params('spikes',True,'sorting')
      }

def mouse_exp_dirs(mouse_num):
    """Return the directories associated with a given mouse num
    
    Only looks in the habituation directories included in config
    """
    mouse_exp_dir_list = []
    for exp_dir in experiment_directories:
        try:
            for exp_session in os.listdir(exp_dir):
                if mouse_num in exp_session:
                    mouse_exp_dir_list.append(os.path.join(exp_dir, exp_session))
        except FileNotFoundError as E:
            print("Could not locate {}, did not search for sessions here.".format(exp_dir))
    return mouse_exp_dir_list

def get_metrics_path(current_dir):
    check_path = os.path.join(current_dir, 'new_metrics.csv')
    try


def copy_files_for_phy(data_path_head, copy_dir):
    try:
        os.mkdir(copy_dir)
    except Exception as E:
        logging.warning(E, exc_info=True)
    for data_file, file_params in data_files.items():
        if file_params.upload==True:
            relpath = relpaths[file_params.relpath]
            source = os.path.join(data_path_head, relpath)
             #subprocess.check_call(['robocopy', '.', 'test', ['test.txt']])
            command_string = "robocopy "+ source +" "+copy_dir +' '+data_file+r" /e /xc /xn /xo"
            try:
                subprocess.check_call(command_string)
            except subprocess.CalledProcessError as E:
                logging.info(E, exc_info=True)

def copy_probe_for_phy(mouse_num, probe, prefix='', copy_dir=data_for_phy_path):
    mouse_exp_dir_list = mouse_exp_dirs(mouse_num)
    sorted_dirs = []
    count = 0
    for mouse_exp_dir in mouse_exp_dir_list:
        for thing in os.listdir(mouse_exp_dir):
            full_path = os.path.join(mouse_exp_dir, thing)
            if os.path.isdir(full_path) and 'sorted' in thing and 'probe'+probe.lower() in thing.lower():
                data_path_head = full_path
                location, session = os.path.split(mouse_exp_dir)
                suffix = location.split('\\')[-1]
                if not(suffix):
                    suffix = location.split('\\')[-2]
                    if not(suffix):
                        suffix = str(count)
                        count+=1
                suffix = '_'+suffix
                copy_session_dir = os.path.join(copy_dir, prefix+session+suffix)
                copy_files_for_phy(data_path_head, copy_session_dir)

def copy_session_for_phy(mouse_num, prefix=''):
    for probe in probe_list:
        copy_probe_for_phy(mouse_num, probe, prefix=prefix)


def open_phy(mouse_num=None, probe=None, plot_distributions=False):
    if mouse_num is None:
        params_path = os.path.join(sorted_data_directory, 'params.py')
        subprocess.Popen(['phy','template-gui',params_path])
    else:
        data_dirs = []
        for data_dir_name in os.listdir(data_for_phy_path):
            if mouse_num in data_dir_name:
                if probe is None or probe in data_dir_name:
                    data_dir = os.path.join(data_for_phy_path, data_dir_name)
                    params_path = os.path.join(data_dir, 'params.py')
                    print(params_path)
                    subprocess.Popen(['phy','template-gui',params_path])
                    if plot_distributions:
                        plot_metric_dir(data_dir)

mouse_list = {
      'Tamina_':  ['404568', '412804'], 
      'Sev_':  ['412792', '412804'], 
      'Josh_':  ['419112', '412804'],
      'Gregg_':  ['408152', '412804'], 
      'Shawn_':  ['425597', '412804'], 
      'Dan D_':  ['403407', '412804'],
      'Ethan_':  ['386129', '412804'], 
      'Leslie_':  ['405751', '412804'], 
      'Corbett':  ['412802', '412804'], 
      'Sam':  ['415148', '412804'], 
      'Nick_':  ['448503', '412804']
    }

if __name__ == '__main__':
    #plot_metric_distributions(sorted_data_directory)
    #open_phy('416356', plot_distributions=True)
    #plt.show()
    for sorter, mice in mouse_list.items():
      for mouse in mice:
        copy_session_for_phy(mouse, sorter)
