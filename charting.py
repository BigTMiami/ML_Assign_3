import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np


def title_to_filename(title, location="Document/figures/working", file_ending="png"):
    safe_title = title.replace(" ", "_")
    safe_title = safe_title.replace(":", "_")
    safe_title = safe_title.replace(",", "_")
    safe_title = safe_title.replace("=", "_")
    safe_title = safe_title.replace("[", "")
    safe_title = safe_title.replace("]", "")
    safe_title = safe_title.replace("(", "")
    safe_title = safe_title.replace(")", "")
    safe_title = safe_title.replace("'", "")
    return f"{location}/{safe_title}.{file_ending}"


def save_to_file(plt, title, location="Document/figures/working"):
    filename = title_to_filename(title, location=location)
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(fname=filename, bbox_inches="tight")


def line_chart(
    x,
    x_label,
    y,
    y_label,
    title,
    sup_title,
):
    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(sup_title, fontsize=16)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    sns.lineplot(x=x, y=y, ax=ax)
    save_to_file(plt, sup_title + " " + title)


def chart_bic_scores(
    scores,
    title,
    sup_title,
):
    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(sup_title, fontsize=16)
    ax.set_title(title)
    for cv_type, score in scores.items():
        sns.lineplot(x=score["k"], y=score["bic_score"], ax=ax, label=cv_type)
    save_to_file(plt, sup_title + " " + title)


def chart_pca_scree(pca, title, sup_title, variance_threshold):
    PC_components = np.arange(pca.n_components_) + 1
    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(sup_title, fontsize=16)
    ax.set_title(title)
    sns.barplot(x=PC_components, y=pca.explained_variance_ratio_)
    sns.lineplot(x=PC_components - 1, y=np.cumsum(pca.explained_variance_ratio_))
    ax.set_ylabel("Explained Variance")
    ax.set_xlabel("PCA Component")
    ax.locator_params(nbins=20, axis="x")
    # plt.xticks(rotation=45, horizontalalignment="right", fontweight="light")
    save_to_file(plt, sup_title + " " + title + f"_vt_{variance_threshold*100:.0f}")
