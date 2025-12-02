# pylint: disable=all

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

### Helper functions for statistics ###


def skewness_and_kurtosis(col):
    skewness_value = round(col.skew(), 2)
    kurtosis_value = round(col.kurt(), 2)

    print(f"Skewness: {skewness_value}")
    print(f"Kurtosis: {kurtosis_value}")


def bool_analysis(col):
    """Function to display basic bool statistics of a boolean column"""
    print("Column type")
    print(col.dtype)

    print("\nUnique values:")
    print(col.unique())

    print("\nTotal amount of unique values:")
    print(col.nunique())

    print("\nDescriptive statistics:")
    print(col.describe())

    print("\nAbsolute Frequency:")
    print(col.value_counts())

    print("\nRelative Frequency (%):")
    print(col.value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

    print("\nImbalance ratio")
    counts = col.value_counts()
    maj, min_ = counts.max(), counts.min()
    imbalance_ratio = maj / min_
    print(f"1 : {imbalance_ratio:.2f} (minority to majority)")


def categorical_analysis(col):
    """Function to display basic statistics of a categorical column"""
    print("Column type")
    print(col.dtype)

    print("\nUnique values:")
    print(list(col.unique()))

    print("\nTotal amount of unique values:")
    print(col.nunique())

    print("\nAbsolute Frequency:")
    print(col.value_counts())

    print("\nRelative Frequency (%):")
    print(col.value_counts(normalize=True).mul(100).round(2).astype(str) + "%")


def numerical_analysis(col):
    """Function to display basic numerical statistics of a numerical column"""
    print("Column type")
    print(col.dtype)

    print("\nMissing values:")
    print(col.isnull().sum())

    print("\nDescriptive statistics:")
    print(col.describe())

    print("\nMeasures of central tendency")
    print("Median:", col.median())
    print("Mode:", col.mode()[0])

    print("\nMeasures of dispersion")
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower fence: {lower_fence}")
    print(f"Upper fence: {upper_fence}")

    outliers = col[(col < lower_fence) | (col > upper_fence)]
    print(f"Number of potential outliers: {outliers.shape[0]}")

    print("\nSkewness and Kurtosis:")
    skewness_and_kurtosis(col)


# def Histogram with KDE

# def boxplot

# def qq plot

##########################################
####### Helper functions for plots #######
##########################################
BASE_DIR = None


def set_base_directory(path):
    """Initialize base directory for saving plots."""
    global BASE_DIR
    BASE_DIR = path
    os.makedirs(BASE_DIR, exist_ok=True)


def pie_plot(col):
    if BASE_DIR is None:
        raise ValueError("BASE_DIR is not set. Use set_base_directory(path) first.")

    filename = f"pieplot_{col.name}.png"
    plot_filename = os.path.join(BASE_DIR, filename)

    if not os.path.isfile(plot_filename):
        trans_freq = col.value_counts()
        explode = [0.02] * len(trans_freq)

        fig = plt.figure(figsize=(5, 5))
        plt.pie(
            trans_freq,
            labels=trans_freq.index,
            autopct="%1.1f%%",
            startangle=0,
            colors=plt.cm.Set2.colors,
            explode=explode,
        )
        plt.title(f"Percentage of {col.name} values")
        plt.tight_layout()
        fig.savefig(plot_filename)
        plt.close(fig)

    return Image(filename=plot_filename)


def bar_plot(col):

    if BASE_DIR is None:
        raise ValueError("BASE_DIR is not set. Use set_base_directory(path) first.")

    filename = f"barplot_{col.name}.png"
    plot_filename = os.path.join(BASE_DIR, filename)

    if not os.path.isfile(plot_filename):
        type_counts = col.value_counts()
        colors_palette = sns.color_palette("husl", len(type_counts))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(type_counts.index, type_counts.values, color=colors_palette, alpha=0.8)
        ax.set_title(f"Distribution of {col.name}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col.name, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        for i, v in enumerate(type_counts):
            ax.text(i, v + v * 0.02, f"{v:,}", ha="center", fontsize=10)

        plt.tight_layout()
        fig.savefig(plot_filename)
        plt.close(fig)

    return Image(filename=plot_filename)


def numerical_plots(col):

    if BASE_DIR is None:
        raise ValueError("BASE_DIR is not set. Use set_base_directory(path) first.")

    filename = f"numerical_plots_{col.name}.png"
    plot_filename = os.path.join(BASE_DIR, filename)

    if not os.path.isfile(plot_filename):
        fig, axes = plt.subplots(3, 1, figsize=(18, 15))

        # 1. Histogram
        axes[0].hist(col, bins=100, edgecolor="black", alpha=0.7, color="skyblue")
        axes[0].set_title(
            f"Histogram: {col.name} Distribution", fontsize=14, fontweight="bold"
        )
        axes[0].set_xlabel(f"{col.name}", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].axvline(
            col.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {col.mean():.1f}",
        )
        axes[0].axvline(
            col.median(),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {col.median():.1f}",
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.5)

        # 2. Box plot
        bp = axes[1].boxplot(col, vert=False, patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        axes[1].set_title(
            f"Box Plot: {col.name} Distribution", fontsize=14, fontweight="bold"
        )
        axes[1].set_xlabel(f"{col.name}", fontsize=12)
        axes[1].set_yticks([])  # No y-ticks needed
        axes[1].grid(True, alpha=0.5, axis="x")

        # 3. Transactions over time
        transactions_per_step = col.value_counts().sort_index()
        axes[2].plot(
            transactions_per_step.index,
            transactions_per_step.values,
            color="navy",
            linewidth=1.5,
        )
        axes[2].set_title(
            f"Transaction Volume Over {col.name}", fontsize=14, fontweight="bold"
        )
        axes[2].set_xlabel(f"{col.name}", fontsize=12)
        axes[2].set_ylabel("Number of Transactions", fontsize=12)
        axes[2].grid(True, alpha=0.5)

        plt.tight_layout()
        fig.savefig(plot_filename)
        plt.close(fig)

    return Image(filename=plot_filename)


def detect_outliers(data):
    # FIXME - numerical types
    numeric_columns = data.select_dtypes(include=["int16", "float32"])

    non_binary_columns = numeric_columns.loc[:, numeric_columns.nunique() > 2]

    Q1 = non_binary_columns.quantile(0.25)
    Q3 = non_binary_columns.quantile(0.75)
    IQR = Q3 - Q1

    def find_outliers(column):
        lower_bound = Q1[column] - 1.5 * IQR[column]
        upper_bound = Q3[column] + 1.5 * IQR[column]
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        print(f"Processing column: {column}")
        if outliers.empty:
            return None, 0.0, None
        else:
            percentage = (len(outliers) / len(data)) * 100
            count_outliers = len(outliers)
            return column, percentage, count_outliers

    columns_with_outliers = [
        find_outliers(column) for column in non_binary_columns.columns
    ]
    columns_with_outliers = [
        (column, percentage, count_outliers)
        for column, percentage, count_outliers in columns_with_outliers
        if column is not None
    ]

    for column, percentage, count_outliers in columns_with_outliers:
        print(
            f"Column: {column}, Percentage of outliers: {percentage:.2f}%, Total number of outliers: {count_outliers}"
        )


def load_from_pickle(file_path):
    """Load data from a pickle file."""
    if os.path.isfile(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")


def save_to_pickle(data, file_path):
    """Save data to a pickle file."""
    with open(file_path, "wb") as file:
        pickle.dump(data, file)
