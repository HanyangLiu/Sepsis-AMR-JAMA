import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_subgroup(prefix='subgroup', postfix='NN', ylim=False):
    df_results = pd.read_csv("../data_analysis/{}_performance_{}.csv".format(prefix, postfix))
    df_results = df_results[df_results["subgroup"] <= 12]
    df = df_results.groupby(['subgroup', 'infection_instance']).first()
    df['AUROC'] = df_results.groupby(['subgroup', 'infection_instance'])['auroc'].mean()
    df['auroc_std'] = df_results.groupby(['subgroup', 'infection_instance'])['auroc'].std()
    df['AUPRC'] = df_results.groupby(['subgroup', 'infection_instance'])['auprc'].mean()
    df['auprc_std'] = df_results.groupby(['subgroup', 'infection_instance'])['auprc'].std()
    df = df.drop(columns=['auroc', 'auprc']).reset_index()

    df = df.rename({
        'subgroup': 'Sub-cohort',
        'infection_instance': 'Sepsis Type',
        'n_instances': '# Instances',
        'positive_rate': 'Positive Rate'
    }, axis='columns')
    df.loc[df['Sepsis Type'] == 'initial', 'Sepsis Type'] = 'Community-acquired'
    df.loc[df['Sepsis Type'] == 'subsequent', 'Sepsis Type'] = 'Hospital-acquired'

    df = df[df['Sepsis Type'].isin(['Community-acquired', 'Hospital-acquired'])]
    df.to_csv("../data_analysis/{}_performance_{}_summarized.csv".format(prefix, postfix), index=False)


    df.loc[df['Positive Rate'] == 0, "AUROC"] = 0
    df.loc[df['Positive Rate'] == 0, "AUPRC"] = 0

    df = df.sort_values(['Sepsis Type', 'Sub-cohort'])
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    sns.barplot(ax=axes[0], data=df, x="Sub-cohort", y="# Instances", hue="Sepsis Type")
    sns.barplot(ax=axes[1], data=df, x="Sub-cohort", y="Positive Rate", hue="Sepsis Type")
    sns.barplot(ax=axes[2], data=df, x="Sub-cohort", y="AUPRC", hue="Sepsis Type")
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in axes[2].patches][:len(df[df["auprc_std"].notna()])]
    y_coords = [p.get_height() for p in axes[2].patches][:len(df[df["auprc_std"].notna()])]
    axes[0].errorbar(x=x_coords, y=y_coords, yerr=0, fmt="none", c="k")
    axes[1].errorbar(x=x_coords, y=y_coords, yerr=0, fmt="none", c="k")
    axes[2].errorbar(x=x_coords, y=y_coords, yerr=df[df["auprc_std"].notna()]["auprc_std"].values, fmt="none", c="k")
    axes[1].legend(loc=2)
    axes[2].legend(loc=2)
    axes[1].set_ylim([0, 0.3])
    if ylim:
        axes[0].set_ylim([0, 3500])
    axes[2].set_ylim([0, 1])
    plt.savefig("plot/{}_performance_{}.pdf".format(prefix, postfix))
    plt.show()


# plot subgroup for NN
plot_subgroup(postfix='NN')
plot_subgroup(prefix='hospital', postfix='NN')
# plot_subgroup(postfix='transfer_hospital_2574')
# plot_subgroup(postfix='transfer_hospital_3148')
# plot_subgroup(postfix='transfer_hospital_6729')

plot_subgroup(postfix='hospital_2574', ylim=True)
plot_subgroup(postfix='hospital_3148', ylim=True)
plot_subgroup(postfix='hospital_6729', ylim=True)

# plot_subgroup(postfix='transfer_test_hospital_3148')
# plot_subgroup(postfix='transfer_test_hospital_6729')

