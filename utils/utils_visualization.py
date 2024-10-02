import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import argparse
import pandas as pd
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import seaborn as sns


def plot_scatter_kde(output_vecs, labels, save_name):
    df_subset = pd.DataFrame(columns=['one', 'two', 'Class'])
    df_subset['one'] = output_vecs[:, 0]
    df_subset['two'] = output_vecs[:, 1]
    # df_subset = df_subset.assign(y=labels.loc[:, 'label'] & (-labels.loc[:, 'if_to_drop']).astype('int').values)
    df_subset = df_subset.assign(y=labels)

    import random
    idx_set = df_subset[df_subset.y == 1].index.tolist()
    rev_set = random.sample(idx_set, round(len(idx_set)/2))
    df_subset.loc[rev_set, 'y'] = 0

    df_subset.loc[df_subset.y != 0, 'Class'] = 'Positive'
    df_subset.loc[df_subset.y == 0, 'Class'] = 'Negative'

    # plot
    sns.set_style('white')
    palette = ["steelblue", "crimson"]

    plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[6.3, 0.7], height_ratios=[0.7, 6.3])

    ax0 = plt.subplot(gs[1, 0])
    sns.set_style('white')
    sns.scatterplot(
        ax=ax0,
        x="one", y="two",
        hue="Class",
        s=25,
        data=df_subset,
        legend="brief",
        palette=sns.color_palette(palette, 2),
        # alpha=0.3
    )
    max1 = df_subset['one'].max()
    min1 = df_subset['one'].min()
    max2 = df_subset['two'].max()
    min2 = df_subset['two'].min()
    ax0.set(xlabel=None)
    ax0.set(ylabel=None)
    ax0.set(xticklabels=[])
    ax0.set(yticklabels=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.set(xlim=[min1 - 0.1*(max1 - min1), max1 + 0.1*(max1 - min1)])
    ax0.set(ylim=[min2 - 0.1 * (max2 - min2), max2 + 0.1 * (max2 - min2)])

    ax1 = plt.subplot(gs[1, 1])
    sns.set_style('white')
    sns.kdeplot(data=df_subset[df_subset.y == 0], y='two', ax=ax1, shade=True, color=palette[0], legend=False)
    sns.kdeplot(data=df_subset[df_subset.y == 1], y='two', ax=ax1, shade=True, color=palette[1], legend=False)
    ax1.set(xticklabels=[])
    ax1.set(yticklabels=[])
    ax1.set(xlabel=None)
    ax1.set(ylabel=None)
    ax0.set(ylim=[min2 - 0.1 * (max2 - min2), max2 + 0.1 * (max2 - min2)])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    ax2 = plt.subplot(gs[0, 0])
    sns.set_style('white')
    sns.kdeplot(data=df_subset[df_subset.y == 0], x='one', ax=ax2, shade=True, color=palette[0], legend=False)
    sns.kdeplot(data=df_subset[df_subset.y == 1], x='one', ax=ax2, shade=True, color=palette[1], legend=False)
    ax2.set(xticklabels=[])
    ax2.set(yticklabels=[])
    ax2.set(xlabel=None)
    ax2.set(ylabel=None)
    ax0.set(xlim=[min1 - 0.1*(max1 - min1), max1 + 0.1*(max1 - min1)])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()


def plot_scatter_kde_2(output_vecs, labels, save_name):
    df_subset = pd.DataFrame(columns=['one', 'two', 'Class'])
    df_subset['one'] = output_vecs[:, 0]
    df_subset['two'] = output_vecs[:, 1]
    df_subset = df_subset.assign(y=labels.loc[:, 'label'].astype('int').values)
    df_subset.loc[df_subset.y != 0, 'Class'] = 'Positive'
    df_subset.loc[df_subset.y == 0, 'Class'] = 'Negative'

    # plot
    sns.set_style('white')
    palette = ["steelblue", "crimson"]

    plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[6.3, 0.7], height_ratios=[0.7, 6.3])

    ax0 = plt.subplot(gs[1, 0])
    sns.set_style('white')
    sns.scatterplot(
        ax=ax0,
        x="one", y="two",
        hue="Class",
        s=25,
        data=df_subset,
        legend="brief",
        palette=sns.color_palette(palette, 2),
        # alpha=0.3
    )
    max1 = df_subset['one'].max()
    min1 = df_subset['one'].min()
    max2 = df_subset['two'].max()
    min2 = df_subset['two'].min()
    ax0.set(xlabel=None)
    ax0.set(ylabel=None)
    ax0.set(xticklabels=[])
    ax0.set(yticklabels=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.set(xlim=[min1 - 0.1*(max1 - min1), max1 + 0.1*(max1 - min1)])
    ax0.set(ylim=[min2 - 0.1 * (max2 - min2), max2 + 0.1 * (max2 - min2)])

    ax1 = plt.subplot(gs[1, 1])
    sns.set_style('white')
    sns.kdeplot(data=df_subset, hue='y', y='two', ax=ax1, shade=True, palette=palette, legend=False)
    ax1.set(xticklabels=[])
    ax1.set(yticklabels=[])
    ax1.set(xlabel=None)
    ax1.set(ylabel=None)
    ax0.set(ylim=[min2 - 0.1 * (max2 - min2), max2 + 0.1 * (max2 - min2)])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    ax2 = plt.subplot(gs[0, 0])
    sns.set_style('white')
    sns.kdeplot(data=df_subset, hue='y', x='one', ax=ax2, shade=True, palette=palette, legend=False)
    ax2.set(xticklabels=[])
    ax2.set(yticklabels=[])
    ax2.set(xlabel=None)
    ax2.set(ylabel=None)
    ax0.set(xlim=[min1 - 0.1*(max1 - min1), max1 + 0.1*(max1 - min1)])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_name)
    # plt.show()


def tSNE_plot(vecs, labels, perplexity, n_iter, save_name):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=2, perplexity=perplexity, n_iter=n_iter)
    output_vecs = tsne.fit_transform(vecs)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    plot_scatter_kde(output_vecs, labels, save_name)


def plot(features):
    from sklearn.manifold import TSNE
    import random
    import time
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import umap.umap_ as umap

    np.random.seed(42)
    idx = np.random.randint(len(features), size=5000)
    vecs = np.concatenate([features[idx, 1, :], features[idx, 0, :]], axis=0)
    mods = np.concatenate([np.zeros((5000,)), np.ones((5000,))])

    # # TSNE
    # tsne = TSNE(n_components=2, verbose=2, perplexity=80, n_iter=5000)
    # output_vecs = tsne.fit_transform(vecs)

    # UMAP
    reducer = umap.UMAP(random_state=42)
    reducer.fit(vecs)
    output_vecs = reducer.transform(vecs)

    df_subset = pd.DataFrame(columns=['one', 'two', 'Class'])
    df_subset['one'] = output_vecs[:, 0]
    df_subset['two'] = output_vecs[:, 1]
    # df_subset = df_subset.assign(y=labels.loc[:, 'label'] & (-labels.loc[:, 'if_to_drop']).astype('int').values)
    df_subset = df_subset.assign(y=mods)

    idx_set = df_subset[df_subset.y == 1].index.tolist()
    rev_set = random.sample(idx_set, round(len(idx_set) / 2))
    df_subset.loc[rev_set, 'y'] = 0

    df_subset.loc[df_subset.y != 0, 'Class'] = 'Positive'
    df_subset.loc[df_subset.y == 0, 'Class'] = 'Negative'

    # plot
    sns.set_style('white')
    palette = ["steelblue", "crimson"]

    plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[6.3, 0.7], height_ratios=[0.7, 6.3])

    ax0 = plt.subplot(gs[1, 0])
    sns.set_style('white')
    sns.scatterplot(
        ax=ax0,
        x="one", y="two",
        hue="Class",
        s=25,
        data=df_subset,
        legend="brief",
        palette=sns.color_palette(palette, 2),
        # alpha=0.3
    )
    max1 = df_subset['one'].max()
    min1 = df_subset['one'].min()
    max2 = df_subset['two'].max()
    min2 = df_subset['two'].min()
    ax0.set(xlabel=None)
    ax0.set(ylabel=None)
    ax0.set(xticklabels=[])
    ax0.set(yticklabels=[])
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.set(xlim=[min1 - 0.1 * (max1 - min1), max1 + 0.1 * (max1 - min1)])
    ax0.set(ylim=[min2 - 0.1 * (max2 - min2), max2 + 0.1 * (max2 - min2)])

    ax1 = plt.subplot(gs[1, 1])
    sns.set_style('white')
    sns.kdeplot(data=df_subset[df_subset.y == 0], y='two', ax=ax1, shade=True, color=palette[0], legend=False)
    sns.kdeplot(data=df_subset[df_subset.y == 1], y='two', ax=ax1, shade=True, color=palette[1], legend=False)
    ax1.set(xticklabels=[])
    ax1.set(yticklabels=[])
    ax1.set(xlabel=None)
    ax1.set(ylabel=None)
    ax0.set(ylim=[min2 - 0.1 * (max2 - min2), max2 + 0.1 * (max2 - min2)])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    ax2 = plt.subplot(gs[0, 0])
    sns.set_style('white')
    sns.kdeplot(data=df_subset[df_subset.y == 0], x='one', ax=ax2, shade=True, color=palette[0], legend=False)
    sns.kdeplot(data=df_subset[df_subset.y == 1], x='one', ax=ax2, shade=True, color=palette[1], legend=False)
    ax2.set(xticklabels=[])
    ax2.set(yticklabels=[])
    ax2.set(xlabel=None)
    ax2.set(ylabel=None)
    ax0.set(xlim=[min1 - 0.1 * (max1 - min1), max1 + 0.1 * (max1 - min1)])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.show()
