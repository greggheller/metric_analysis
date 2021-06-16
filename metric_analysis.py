import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as kde
import logging

from datetime import datetime as dt

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_metrics_path(current_dir):
    check_path = os.path.join(current_dir, 'new_metrics.csv')
    try:
      csv_path = glob.glob(str(check_path))[0]
    except IndexError as E:
        check_path = os.path.join(current_dir, '*metrics.csv')
        csv_path = glob.glob(str(check_path))[0]
    return csv_path



def get_subplot_dim(n):
    i = 0
    while i*i < n:
        i+=1
    mid = (i-1)*i
    upper = i**2
    #print('lower ', lower, ' upper ', upper, ' mid ', mid)
    if n <= mid:
      return i-1, i
    else:
      return i, i

def plot_metric_dir(dirpath):
    csv_path = get_metrics_path(dirpath)
    plot_metric_csv(csv_path)

def plot_metric_csv(csv_path):
    with open(csv_path, mode='r') as csv_file:
        df = pandas.read_csv(csv_file, index_col=0)
    plot_metric_distributions(df)

def plot_metric_distributions(dataframe, include_columns=None):
    if include_columns is None:
        include_set = {
          'isolation_distance',
          'log_isolation_distance',
          'silhouette_score',
          'd_prime',
          'snr',
          'nn_hit_rate',
          'presence_ratio', 
          'max_drift',  
         'cumulative_drift',
          'amplitude_cutoff', 
          'l_ratio',
          'log_l_ratio',
          'nn_miss_rate',
          'isi_violations',
          'log_isi_viol'
        }
    else:
        include_set = set(include_columns)

    include_list = list(include_set.intersection(set(dataframe.columns)))
    df = dataframe.loc[:, include_list]

    fig_num = 0
    print(df)
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    try:
        #plt.figure()
        #sns.pairplot(data=df)
        pass
    except Exception as E:
      print('failed to produce pairplot')

    
    #sns.violinplot(df)
    violin_num = len(df.columns)
    subplot_x, subplot_y = get_subplot_dim(violin_num)
    print(subplot_x, subplot_y)
    fig, axes = plt.subplots(subplot_x, subplot_y)
    x_idx, y_idx = (0,0)
    for column in df.columns:
      print(column)
      if column in include_list:
          print(column)
          print(x_idx, y_idx)
          plt.subplot(axes[x_idx, y_idx])
          try:
              sns.violinplot(x=df.loc[:, column])
              if (x_idx + 1 < subplot_x):
                x_idx += 1
              else:
                x_idx = 0
                y_idx += 1
          except Exception as E:
              print('failed to produce violinplot for ' + column)
              logging.error(E, exc_info=True)
    plt.tight_layout()
    #plt.show()


        
    
def josh_contuour(df, x_col_name, y_col_name):
    from fastkde import fastKDE
    plt.figure()
    x = df.loc[:, x_col_name]
    y = df.loc[:, y_col_name]

    N = pow(2,8) + 1

    myPDF,axes = fastKDE.pdf(x,y, axes=(np.linspace(0,10,N), np.linspace(0,1,N)))

    plt.clf()

    #Extract the axes from the axis list
    v1,v2 = axes
    myPDF.shape

    plt.imshow(myPDF,aspect='auto', origin='lower', vmax=3,cmap='Purples')

    a = np.argmax(myPDF,1)
    plt.plot(a,np.arange(N),color='slategrey')

    #plt.xlim([0,15])
    #plt.ylim([0,1])
    # %%
    nbins=60
    k = kde.gaussian_kde([x,y])
    x_support = np.linspace(0,np.max(x),nbins)
    y_support = np.linspace(0,np.max(y),nbins)
    xx, yy = np.meshgrid(x_support, y_support)
    z = k([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.clf()
    plt.pcolormesh(xx, yy, z) #.reshape(xi.shape))
    plt.xlabel(x_col_name)
    plt.ylabel(y_col_name)





def plot_grid(df, x_col_names, y_col_names, plot_function):
    x_col_names = list(set(x_col_names).intersection(set(df.columns)))
    y_col_names = list(set(y_col_names).intersection(set(df.columns)))
    #plt.figure()
    subplot_idx = 1
    print(x_col_names)
    print(y_col_names)
    for x_col_name in x_col_names:
        for y_col_name in y_col_names:
            #plt.subplot(len(x_col_names), len(y_col_names), subplot_idx)
            print('plotting '+ y_col_name +' against '+x_col_name)
            try:
                ax = clean_plot(df, x_col_name, y_col_name, plot_function)
                #ax.set_title(y_col_name)
                #ax.set_xlabel(x_col_name)
            except Exception as E:
                print(E)
            plt.clf()
            subplot_idx += 1


def clean_plot(df_in, x_col_name, y_col_name, plot_function):
    df = df_in.copy()
    #df.dropna(subset=[x_col_name, y_col_name], how='all')
    ok_units = (df['silhouette_score'] > -1) & \
                   (df['isolation_distance'] < 1e3) & \
                   (df['amplitude_cutoff'] < 0.5) & \
                   (df['nn_hit_rate'] > 0)
    df = df[ok_units]
    if x_col_name == y_col_names:
      df = df.loc[:, [x_col_name]]
    else:
      df = df.loc[:, [x_col_name, y_col_name]]
    df.replace([np.inf, -np.inf], np.nan)
    try:
      df[np.isinf(df)] = np.nan
      df.dropna(how='any', inplace=True)
      print(df)
      print(np.isnan(df).any())
      print(np.isnan(df).any())
      print(np.isinf(df).any())
      print(np.isinf(df).any())
    except Exception as E:
      print(E)
    try:
      ax = plot_function(df, x_col_name, y_col_name)
    except Exception as E:
      print(E)
    plt.savefig(y_col_name +' against '+x_col_name+'_'+plot_function.__name__)
    return ax

def contour(df, x_col_name, y_col_name):
    ax = sns.jointplot(data=df, x=x_col_name, y=y_col_name, kind="kde", n_levels=30)
    return ax

def correlation_matrix(dataframe, x_col_names, y_col_names, colormap='Blues'):
    
    
    df_corr = dataframe.corr()
    df = df_corr.loc[y_col_names, x_col_names]
    my_heatmap(df, x_col_names, y_col_names, colormap)

def my_heatmap(df, x_col_names, y_col_names, colormap='Blues', filename=None):
    f = plt.figure(figsize=(19, 15))
    ax = plt.gca()
    ax = sns.heatmap(df, cmap=colormap, vmin=-1, vmax=1)
    #im = ax.matshow(df, cmap=colormap)
    
    ax.set_xticks(np.arange(len(x_col_names))+.5)
    ax.set_xticklabels(x_col_names)
    ax.set_xlim(-.5, len(x_col_names))
    ax.set_yticks(np.arange(len(y_col_names))+.5)
    ax.set_yticklabels(y_col_names)
    ax.set_ylim(len(y_col_names), -.5 )

    # Set ticks on both sides of axes on
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    # Rotate and align bottom ticklabels
    plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45,
             ha="right", va="center", rotation_mode="anchor")
    # Rotate and align top ticklabels
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="left", va="center",rotation_mode="anchor")

    if filename is None:
        filename = dt.now().strftime('%H_%M_%S')

    plt.savefig(filename, format='pdf')
    #cb = f.colorbar(im)
    #plt.title('Correlation Matrix', fontsize=16);

    # Set ticks on both sides of axes on
    #cb.ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # Rotate and align bottom ticklabels
    #plt.setp([tick.label1 for tick in cb.ax.xaxis.get_major_ticks()], rotation=45,
    #     ha="right", va="center", rotation_mode="anchor")
x_col_names = [
      #'isolation_distance',

      #'log2_isolation_distance',
      #'log100_isolation_distance',  
      'firing_rate', 
      'presence_ratio',   
      'log_isi_viol',     
      'nn_hit_rate',
      'log_isolation_distance',
                'log_l_ratio',
      'silhouette_score',
      'snr',
      'd_prime',
      'amplitude_cutoff', 

      'max_drift',  
     'cumulative_drift',
      'nn_miss_rate',

      #'isi_violations',
      #'log_presence_ratio', 

      #'l_ratio',

      #'log2_l_ratio',
      #'log100_l_ratio',
      #'log2_isi_viol',
      #'log100_isi_viol'
    ]
y_col_names = [
  'firing_rate',
  'sustained_idx_fl',
  'time_to_peak_fl',
  'run_mod_dg',

  'g_dsi_dg',
  'g_osi_dg',
  'pref_ori_multi_dg',
  'pref_tf_multi_dg',
  'time_to_first_spike_fl',

  'area_rf',
  'time_to_peak_rf',
  'elevation_rf',

  'on_screen_rf',
  'p_value_rf',
  'mod_idx_dg',

  'pref_ori_dg',

  'pref_tf_dg',



]
y_col_names_all = []
y_col_names_all.extend(y_col_names)
y_col_names_all.extend(x_col_names)

inverted = [
      'max_drift',  
     'cumulative_drift',
      'nn_miss_rate',
      'log_l_ratio',
      'amplitude_cutoff', 
      'log_isi_viol',
]

def prepare_metric_df(df):
    try:
        df.rename(columns={'isi_viol':'isi_violations'}, inplace=True)
    except Exception as E:
        print(e)
    print(df.columns)
    df['log2_isolation_distance'] = np.log10(np.log10(df.loc[:, 'isolation_distance']+1)+1)
    df['log2_l_ratio'] = np.log10(np.log10(df.loc[:, 'l_ratio']+1)+1)
    df['log2_isi_viol'] = np.log10(np.log10(df.loc[:, 'isi_violations']+1)+1)
    df['log_isolation_distance'] = np.log10(df.loc[:, 'isolation_distance'])
    df['log_l_ratio'] = np.log10(df.loc[:, 'l_ratio'])
    df['log_isi_viol'] = np.log10(df.loc[:, 'isi_violations'])
    df['log100_isolation_distance'] = np.log10((1-df.loc[:, 'isolation_distance'])*100)
    df['log100_l_ratio'] = np.log10((1-df.loc[:, 'l_ratio'])*100)
    df['log100_isi_viol'] = np.log10((1-df.loc[:, 'isi_violations'])*100)
    df['log_presence_ratio'] = np.log10((1-df.loc[:, 'presence_ratio'])*100)
    df.replace([np.inf, -np.inf], np.nan)

    for col in inverted:
      df[col] = -df[col]
    return df

if __name__ == '__main__':

    metrics_412804 = r"C:\Users\svc_neuropix\Documents\metric_analysis\412804\metrics.csv"
    all_units = r"C:\Users\svc_neuropix\Documents\metric_analysis\unit_table_20190927.csv"
    #plot_metric_csv(metrics_412804)
    csv_path = all_units
    with open(csv_path, mode='r') as csv_file:
        df = pandas.read_csv(csv_file, index_col=0)
    df = prepare_metric_df(df)

    
    #plot_grid(df, x_col_names, y_col_names, josh_contuour)
    #plot_grid(df, x_col_names, y_col_names_all, josh_contuour)
    #plot_grid(df, x_col_names, y_col_names, contour)
    #plot_grid(df, x_col_names, y_col_names_all, contour)
    #correlation_matrix(df, x_col_names, x_col_names, colormap='RdBu_r')
    #correlation_matrix(df, x_col_names, y_col_names, colormap='RdBu_r')
    #plot_metric_distributions(df)
    #clean_plot(df, 'log_isi_viol', 'firing_rate', contour)
    #clean_plot(df, 'log_isi_viol', 'g_dsi_dg', contour)
    #clean_plot(df, 'log_isi_viol', 'g_osi_dg', contour)
    #clean_plot(df, 'log_isi_viol', 'time_to_first_spike_fl', contour)
    #clean_plot(df, 'log_isi_viol', 'sustained_idx_fl', contour)


    #clean_plot(df, 'amplitude_cutoff', 'firing_rate', contour)
    #clean_plot(df, 'amplitude_cutoff', 'g_dsi_dg', contour)
    #clean_plot(df, 'amplitude_cutoff', 'g_osi_dg', contour)
    #clean_plot(df, 'amplitude_cutoff', 'time_to_first_spike_fl', contour)
    #clean_plot(df, 'amplitude_cutoff', 'sustained_idx_fl', contour)


    #clean_plot(df, 'log_isolation_distance', 'firing_rate', contour)
    #clean_plot(df, 'log_isolation_distance', 'g_dsi_dg', contour)
    #clean_plot(df, 'log_isolation_distance', 'g_osi_dg', contour)
    #clean_plot(df, 'log_isolation_distance', 'time_to_first_spike_fl', contour)
    #clean_plot(df, 'log_isolation_distance', 'sustained_idx_fl', contour)

    #clean_plot(df, 'presence_ratio', 'firing_rate', contour)
    #clean_plot(df, 'presence_ratio', 'g_dsi_dg', contour)
    #clean_plot(df, 'presence_ratio', 'g_dsi_dg', contour)
    #clean_plot(df, 'presence_ratio', 'time_to_first_spike_fl', contour)
    #clean_plot(df, 'presence_ratio', 'sustained_idx_fl', contour)


    clean_plot(df, 'snr', 'firing_rate', contour)
    clean_plot(df, 'snr', 'g_dsi_dg', contour)
    clean_plot(df, 'snr', 'g_dsi_dg', contour)
    clean_plot(df, 'snr', 'time_to_first_spike_fl', contour)
    clean_plot(df, 'snr', 'sustained_idx_fl', contour)


    #clean_plot(df, 'd_prime', 'nn_hit_rate', josh_contuour)
    plt.show()