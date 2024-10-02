import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import scipy
os.chdir("../")


def plot_corr(df_results, label):
    df_results = df_results[np.logical_and(df_results["infection_instance"] == 'all', df_results["hospital_id"] == 'all')]
    df = df_results.groupby(['subgroup', 'infection_instance']).first()
    df['AUROC'] = df_results.groupby(['subgroup', 'infection_instance'])['auroc'].mean()
    df['auroc_std'] = df_results.groupby(['subgroup', 'infection_instance'])['auroc'].std()
    df['AUPRC'] = df_results.groupby(['subgroup', 'infection_instance'])['auprc'].mean()
    df['auprc_std'] = df_results.groupby(['subgroup', 'infection_instance'])['auprc'].std()
    df = df.drop(columns=['auroc', 'auprc']).reset_index()

    plt.figure()
    ax = sns.regplot(data=df, x="positive_rate", y="AUPRC", ci=95)
    plt.xlabel('Positive Rate', fontsize=13)
    plt.ylabel('AUPRC', fontsize=13)
    r, p = scipy.stats.pearsonr(df["positive_rate"], df["AUPRC"])
    blue_line = Line2D([0], [0], label="Correlation R={0:.2f}, p-value={1:.3f}".format(r, p) if p < 0.01
                       else "Correlation R={0:.2f}, p-value={1:.2f}")
    point = Line2D([0], [0], label=label, marker='o', linestyle='')
    plt.legend(handles=[blue_line, point])
    plt.savefig("plot/positive_rate_{}.pdf".format(label))
    plt.show()

    plt.figure()
    ax = sns.regplot(data=df, x="fraction_in_set", y="AUPRC", ci=95)
    plt.xlabel('Sample Size (Fraction in Dataset)', fontsize=13)
    plt.ylabel('AUPRC', fontsize=13)
    r, p = scipy.stats.pearsonr(df["fraction_in_set"], df["AUPRC"])
    blue_line = Line2D([0], [0], label="Correlation R={0:.2f}, p-value={1:.3f}".format(r, p) if p < 0.01
                       else "Correlation R={0:.2f}, p-value={1:.2f}".format(r, p))
    point = Line2D([0], [0], label=label, marker='o', linestyle='')
    plt.legend(handles=[blue_line, point])
    plt.savefig("plot/sample_size_{}.pdf".format(label))
    plt.show()


df_results = pd.read_csv("../data_analysis/subgroup_performance_NN.csv")
df_results = df_results[df_results.subgroup <= 12]
# df_results = df_results[df_results.fraction_in_set < 0.2]
plot_corr(df_results, label="Subgroup")

df_results = pd.read_csv("../data_analysis/hospital_performance_NN.csv")
plot_corr(df_results, label="Hospital")