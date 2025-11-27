# pylint: disable=all

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

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


### Helper functions for plots ###


def pie_plot(col):
    trans_freq = col.value_counts()
    explode = [0.02] * len(trans_freq)

    plt.figure(figsize=(5, 5))
    plt.pie(
        trans_freq,
        labels=trans_freq.index,
        autopct="%1.1f%%",
        startangle=0,
        colors=plt.cm.Set2.colors,
        explode=explode,
    )
    plt.title(f"Percentatge of {col.name} values")
    plt.show()


def bar_plot(col):
    type_counts = col.value_counts()
    colors_palette = sns.color_palette("husl", len(type_counts))

    plt.figure(figsize=(10, 6))
    ax = type_counts.plot(kind="bar", color=colors_palette, alpha=0.8)
    ax.set_title(f"Distribution of {col.name}", fontsize=14, fontweight="bold")
    ax.set_xlabel(col.name, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(type_counts):
        ax.text(i, v + v * 0.02, f"{v:,}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()


def detect_outliers(data):
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
