# pylint: disable=all

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns


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


def skewness_and_kurtosis(df, column_name):
    skewness_value = df[column_name].skew()
    kurtosis_value = df[column_name].kurt()

    return {"skewness": skewness_value, "kurtosis": kurtosis_value}


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
