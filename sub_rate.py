import os
import pandas as pd
import metric_analysis as ma

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics as sm
from colour import Color

q_tsv_dir = r"C:\Users\greggh\Documents\Python Scripts\metrics_analysis\412804_subjective_ratings"

metric_csv_path = r"C:\Users\greggh\Documents\Python Scripts\metrics_analysis\412804\metrics.csv"


new_metric_path = os.path.join(os.path.split(metric_csv_path)[0], 'metric_with_subjective.csv')

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def add_q_column(metric_df, q_tsv_path, suffix=None):
    with open(q_tsv_path, mode='r') as csv_file:
        qdf = pd.read_csv(csv_file, sep='\t')
    if not(suffix is None):
        qdf.rename(columns={'q': 'grader_'+str(suffix)}, inplace=True)
    print(qdf.head())
    #print(qdf.columns)
    #print(metric_df.columns)
    df_outer = pd.merge(metric_df, qdf, on='cluster_id', how='outer')
    return df_outer

def generate_sub_csv():
    with open(metric_csv_path, mode='r') as csv_file:
        metric_df = pd.read_csv(csv_file, index_col=0)
    i=0 
    for q_tsv in os.listdir(q_tsv_dir):
        i+=1
        q_tsv_path = os.path.join(q_tsv_dir, q_tsv)
        metric_df = add_q_column(metric_df, q_tsv_path, i)

    add_sdk_pass_column(metric_df)
    metric_df.to_csv(new_metric_path)

def get_qdf(metric_df):
    qdf = metric_df.loc[:, metric_df.columns.str.contains('grader_')]
    return qdf


def grouped_bar(df, x_label, y_label, sub_sub_group_count=1, change_color_between_groups=False, stacked=False, override_color=False):
    # set width of bar
    
     
    # Set position of bar on X axis
    subgroup_count = df.shape[0]
    print('subgroup_count', subgroup_count)
    #bar_count = 
    print('The dataframe looks like:')
    print(df.head())
    r1 = np.arange(subgroup_count)
    barWidth = .6/((df.shape[1]))
     
    # Make the plot
    f = plt.figure(figsize=(19, 15))
    green = Color("green")
    if change_color_between_groups:
        colors_needed = df.shape[0]/(sub_sub_group_count)+1
    else:
        colors_needed = df.shape[1]/(sub_sub_group_count)+1
    if stacked:
        colors_needed = (df.shape[0]/(sub_sub_group_count)+1)+1
    #print(subgroup_count, sub_sub_group_count, colors_needed)
    colors = list(green.range_to(Color("blue"),colors_needed))
    #print(len(colors))
    bottoms=np.zeros(len(df.columns))
    for idx, column in enumerate(df):
        if stacked:
            r = [1 + (barWidth)*idx for x in r1]
        else:
            r = [x+(barWidth)*idx for x in r1]
        try:
            color = str(colors[int(idx/sub_sub_group_count)])
            if override_color:
                color=override_color
        except IndexError as E:
            color=None
        for i, value in enumerate(df[column]):
            print('value: ', value)
            if change_color_between_groups:
                color = str(colors[int(i/sub_sub_group_count)])
            if stacked:
                plt.bar(r[i], value, bottom=bottoms[idx], color=color, width=barWidth, edgecolor='white', label=column)
                bottoms[idx] = bottoms[idx] + value
            else:
                plt.bar(r[i], value, color=color, width=barWidth, edgecolor='white', label=column)
        #plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')
        #plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
     
    # Add xticks on the middle of the group bars
    if stacked:
        plt.xlabel(x_label, fontweight='bold')
        plt.ylabel(y_label, fontweight='bold')
        #if change_color_between_groups:
        plt.xticks([1+r*barWidth for r in range(len(df.columns))], df.columns)
    else:
        plt.xlabel(x_label, fontweight='bold')
        plt.ylabel(y_label, fontweight='bold')
        #if change_color_between_groups:
        plt.xticks([r + barWidth*(df.shape[1]-1)/2 for r in range(df.shape[0])], df.index)
    #else:
        #plt.xticks([r + barWidth*(df.shape[1]+2)/2 for r in range(df.shape[0])], df.index)

    # Create legend & Show graphic
    #plt.legend()
    name = x_label+'_and_'+y_label+'.pdf'
    name = name.replace(" ", "")
    name = name.replace(")", "")
    name = name.replace("(", "")
    name = name.replace("/", "")
    name = name.replace("+", "")
    plt.savefig(name, format='pdf')

def add_sdk_pass_column(df):
    df['sdk_pass'] = (df['presence_ratio']>.95) & (df['isi_viol']<.5) & (df['amplitude_cutoff']<.1)


def rating_counts(qdf_in, sub_sub_group_count):
    #qdf = get_qdf(metric_df)
    #print(qdf.shape)
    #qdf.replace('', np.nan, inplace=True)
    #for col in qdf:
    #    sns.distplot(col)
    #plt.show()
    #qdf.dtypes
    print('QDF at this point looks like')
    print(qdf_in)
    qdf=qdf_in.copy()
    counts = qdf.apply(pd.value_counts)
    #print(counts.index)
    grouped_bar(counts, x_label='Rating', y_label='Counts', sub_sub_group_count=sub_sub_group_count)
    #plt.legend(loc='upper left', ncol=2)

if __name__ == '__main__':
    #"""
    with open(new_metric_path, mode='r') as csv_file:
        metric_df = pd.read_csv(csv_file, index_col=0)
    
    #"""
    qdf = get_qdf(metric_df)
    #print(qdf)
    
    qdf['sdk_pass']=metric_df['sdk_pass']
    qdf.replace('', np.nan, inplace=True)
    qdf.dropna(inplace=True)
    sdk_pass = qdf.loc[:, 'sdk_pass']
    qdf = qdf.drop('sdk_pass', 1)
    qdf['Average']=np.round(qdf.mean(axis=1))
    metric_df['Average'] = qdf['Average']
    compare_cutoffs = [1,2,3,4,5]
    index_list = [str(x) for x in compare_cutoffs]
    index_list_f = ['>'+str(x) for x in [1,2,3,4]]
    cutoff_column_list = []
    for column in qdf:
        cutoff_column_list.extend([column+'_recall'])#, column+'_recall', column+'_fscore'])
    sorter_v_cutoff = pd.DataFrame(0, index=index_list, columns=cutoff_column_list)
    sorter_v_cutoff_f = pd.DataFrame(0, index=index_list_f, columns=['Average'])
    y_pred = sdk_pass
    #print(sum(y_pred))
    print(qdf)
    print(y_pred)
    

    for column in qdf:
        for cutoff in compare_cutoffs:
            index = str(cutoff)
            columns = column+'_recall'
            y_true = qdf[column]>cutoff
            #y_pred = [True, True, True, True, True, False, False]
            #y_true = [True, True, True, True, True, True, False]
            precision, recall, fscore, support = sm.precision_recall_fscore_support( y_true, y_pred, average='binary', pos_label=True)
            #print(y_pred)
            #print(y_true)
            #print(precision, recall, fscore, support)
            #riast(value_error)
            mask =  qdf[column]==cutoff
            print(mask)
            print(sdk_pass[mask])
            print(sum(sdk_pass[mask]))
            print(sum(mask))
            fraction = float(sum(sdk_pass[mask]))/sum(mask)
            print(fraction)
            sorter_v_cutoff.loc[index, columns]= fraction
            if column == 'Average' and cutoff<5:
                index = '>'+str(cutoff)
                sorter_v_cutoff_f.loc[index, 'Average'] = fscore
    print(sorter_v_cutoff)
    grouped_bar(sorter_v_cutoff, x_label='Rating Threshold', y_label='Fraction above threshold', sub_sub_group_count=1)
    #print(sorter_v_cutoff_f.shape[1])
    #print(sorter_v_cutoff_f)
    
    grouped_bar(sorter_v_cutoff_f, x_label='Rating Threshold', y_label='F Score', sub_sub_group_count=1, override_color='steelblue')
    #sns.barplot(data=sorter_v_cutoff_f)
    #plt.savefig('sns_bar.pdf')

    for column in qdf:
        for cutoff in compare_cutoffs:
            index = '>'+str(cutoff)
            columns = column+'_recall'
            y_true = qdf[column]
            #y_pred = [True, True, True, True, True, False, False]
            #y_true = [True, True, True, True, True, True, False]
            #precision, recall, fscore, support = sm.precision_recall_fscore_support( y_true, y_pred, average='binary', pos_label=False)
            #print(y_pred)
            #print(y_true)
            #print(precision, recall, fscore, support)
            #riast(value_error)
            sorter_v_cutoff.loc[index, columns] = recall
    #grouped_bar(sorter_v_cutoff, x_label='Rating Threshold', y_label='Inverse Recall (tn/(tn+fp))', sub_sub_group_count=1)
    #print(qdf)
    #raise(ValueError)
    sorter_columns_list = []
    for i in range(len(qdf.columns)):
        i_str = str(i)
        sorter_columns_list.append(i_str+'_fscore')#[i_str+'_precision', i_str+'_recall', i_str+'_fscore'])
    sorter_v_sorter = pd.DataFrame(0, index=qdf.columns, columns=sorter_columns_list)
    for column in qdf:
        y_true = qdf[column]
        for i , column2 in enumerate(qdf):
            i_str = str(i)
            index = column
            columns = i_str+'_fscore'
            #if column == column2:
            #    sorter_v_sorter.loc[index, columns] = 0
            #else:
            y_pred = qdf[column2]
            #print(y_true)
            #print(y_pred)
            #print(y_pred==y_true)
            precision, recall, fscore, support = sm.precision_recall_fscore_support(y_true, y_pred, average='macro')
            #print(precision, recall, fscore, support )
            sorter_v_sorter.loc[index, columns] = fscore
    #grouped_bar(sorter_v_sorter, x_label='Grader', y_label='F Score')
    #ma.correlation_matrix(qdf, qdf.columns, qdf.columns, colormap='RdBu_r')
    #plt.title('correlation between graders')
    ma.my_heatmap(sorter_v_sorter, qdf.columns, qdf.columns, colormap='RdBu_r')
    #plt.title('F score between graders')

    metric_df = ma.prepare_metric_df(metric_df)
    ma.correlation_matrix(metric_df, qdf.columns, ma.x_col_names, colormap='RdBu_r')
    
    rating_counts(qdf, sub_sub_group_count=1)

    print(qdf.columns)
    qdf.drop('Average', axis=1, inplace=True)
    print(qdf.columns)
    for column in qdf:
        col_diffs = None
        for column2 in qdf:
            if column == column2:
                pass
            else:
                if col_diffs is None:
                    col_diffs = qdf[column]-qdf[column2]
                else:
                    col_diffs= pd.concat([col_diffs, (qdf[column]-qdf[column2])])
        #print(col_diffs)
        try:
            sorter_diffs = pd.concat([sorter_diffs, col_diffs], axis=1)
        except NameError as E:
            sorter_diffs = col_diffs
    #print(sorter_diffs)
    counts = sorter_diffs.apply(pd.value_counts)
    #counts = np.log10(counts)
    total = len(sorter_diffs.index)
    counts = counts/total
    counts.columns = qdf.columns
    counts = counts/len(counts.columns)
    grouped_bar(counts.T, x_label='Grader', y_label='Fraction of total comparisons', stacked=True, sub_sub_group_count=1, change_color_between_groups=True)

    #print(metric_df.columns)
    
    #generate_sub_csv()
   
    plt.show()