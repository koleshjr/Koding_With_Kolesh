import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import textwrap
from tqdm import tqdm


import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = 2200
pd.options.display.max_rows = 2200

#### Data Loading Utils and Exploration
"<<<<<<<<<<<<<------------------------ Loading Datasets and Exploring Dataset Information ------------------------------------->>>>>>>>>>>>>"
def loadDataset(path: str, dataset: str)-> pd.DataFrame:
    '''
     Takes in the path of your working dir and the name of your dataset
     this is not all the files we can use but you can add them with time but has most of the used dataset types
    
    '''
    if dataset.endswith('.csv'):
        data = pd.read_csv(path + dataset)
    elif dataset.endswith('.json'):
        data = pd.read_json(path + dataset)
    elif dataset.endswith('.parquet'):
        data = pd.read_parquet(path + dataset)
    elif dataset.endswith('.xlsx'):
        data = pd.read_excel(path + dataset)
    elif dataset.endswith('.feather'):
        data = pd.read_feather(path  + dataset)
    return data






def datasetInfo(data: pd.DataFrame, id_column: str, target_column: str):
    """
    Display comprehensive information about a pandas DataFrame.

    Arguments:
    data -- the DataFrame to display information for
    
    Returns:
    categorical_cols -- list of categorical column names
    numerical_cols -- list of numerical column names
    """
    
    # Dataset Overview
    print("Dataset Overview")
    print("----------------")
    print(f"Shape: {data.shape}\n")
    
    # Column Names
    print("Column Names")
    print("------------")
    print(data.columns.tolist())
    print()
    
    # Data Types
    print("Data Types")
    print("----------")
    print(data.dtypes)
    print()
    
    # Missing Values
    print("Missing Values")
    print("--------------")
    print(data.isnull().sum().sort_values(ascending=False))
    print()
    
    # Descriptive Statistics
    print("Descriptive Statistics")
    print("-----------------------")
    display(data.describe().T.sort_values(by='std', ascending=False)
            .style.background_gradient(cmap='GnBu')
            .bar(subset=["max"], color='#BB0000')
            .bar(subset=["mean"], color='green'))
    print()
    

    # Categorical and Numerical Columns
    categorical_cols = data.select_dtypes(include='object').columns.tolist()
    numerical_cols = data.select_dtypes(exclude='object').columns.tolist()
    
    
    categorical_cols = [col for col in categorical_cols if col not in [id_column,target_column]]
    numerical_cols = [col for col in numerical_cols if col not in [id_column, target_column]]

    
    return categorical_cols, numerical_cols, target_column, id_column

def findDifferentialInfo(train, test, __featToExcl=[]):
    '''
    Describe data and difference between train and test datasets.
    
    Arguments:
    train -- the training dataset (pandas DataFrame)
    test -- the test dataset (pandas DataFrame)
    __featToExcl -- list of features to exclude from analysis (default: [])
    target_for_vcramer -- the target variable for calculating Cramer's V coefficient (default: None)
    
    Returns:
    df_stats -- DataFrame containing statistics and differences between train and test datasets
    
    '''
    
    stats = []
    __featToAnalyze = [v for v in list(train.columns) if v not in __featToExcl]

    for col in tqdm(__featToAnalyze):

        dtrain = dict(train[col].value_counts())
        dtest = dict(test[col].value_counts())

        set_train_not_in_test = set(dtest.keys()) - set(dtrain.keys())
        set_test_not_in_train = set(dtrain.keys()) - set(dtest.keys())

        dict_train_not_in_test = {key: value for key, value in dtest.items() if key in set_train_not_in_test}
        dict_test_not_in_train = {key: value for key, value in dtrain.items() if key in set_test_not_in_train}

        nb_moda_test, nb_var_test = len(dtest), pd.Series(dtest).sum()
        nb_moda_abs, nb_var_abs = len(dict_train_not_in_test), pd.Series(dict_train_not_in_test).sum()
        nb_moda_train, nb_var_train = len(dtrain), pd.Series(dtrain).sum()
        nb_moda_abs_2, nb_var_abs_2 = len(dict_test_not_in_train), pd.Series(dict_test_not_in_train).sum()

        stats.append((col, train[col].nunique()
                      , str(nb_moda_abs) + '   (' + str(round(100 * nb_moda_abs / nb_moda_test, 1)) + '%)'
                      , str(nb_moda_abs_2) + '   (' + str(round(100 * nb_moda_abs_2 / nb_moda_train, 1)) + '%)'
                      , str(train[col].isnull().sum()) + '   (' + str(
            round(100 * train[col].isnull().sum() / train.shape[0], 1)) + '%)'
                      , str(test[col].isnull().sum()) + '   (' + str(
            round(100 * test[col].isnull().sum() / test.shape[0], 1)) + '%)'
                      , str(round(100 * train[col].value_counts(normalize=True, dropna=False).values[0], 1))
                      , train[col].dtype))

    df_stats = pd.DataFrame(stats, columns=['Feature'
        , 'Unique values (train)', "Unique values in test not in train (and %)"
        , "Unique values in train not in test (and %)"
        , 'NaN in train (and %)', 'NaN in test (and %)', '% in the biggest cat. (train)'
        , 'dtype'])

    return df_stats






def columnWiseNullDistributionComparison(train:pd.DataFrame, test:pd.DataFrame):
    '''
    A function that takes in the train and test dataframes and plots column wise Null Distribution Comparison
    
    Arguments:
        train: pd.DataFrame
        test : pd.DataFrame
    
    '''
    test_null = pd.DataFrame(test.isna().sum())
    test_null = test_null.sort_values(by = 0 ,ascending = False)[:-5]
    train_null = pd.DataFrame(train.isna().sum())
    train_null = train_null.sort_values(by = 0 ,ascending = False)[:-6]
    fig, axes = plt.subplots(1,2, figsize=(18,10))
    sns.barplot( y =test_null.index ,  x  = test_null[0] ,ax = axes[1] ,palette = "viridis")
    sns.barplot( y =train_null.index ,  x  = train_null[0],ax = axes[0],palette = "viridis")
    axes[0].set_xlabel("TRAIN DATA COLUMNS")
    axes[1].set_xlabel("TEST DATA COLUMNS");

def rowWiseNullDistributionComparison(train:pd.DataFrame, test:pd.DataFrame):
    '''
    A function that takes in the train and test dataframes and plots column wise Null Distribution Comparison
    
    Arguments:
        train: pd.DataFrame
        test : pd.DataFrame
    
    '''

    missing_train_row = train.isna().sum(axis=1)
    missing_train_row = pd.DataFrame(missing_train_row.value_counts()/train.shape[0]).reset_index()
    missing_test_row = test.isna().sum(axis=1)
    missing_test_row = pd.DataFrame(missing_test_row.value_counts()/test.shape[0]).reset_index()
    missing_train_row.columns = ['no', 'count']
    missing_test_row.columns = ['no', 'count']
    missing_train_row["count"] = missing_train_row["count"]*100
    missing_test_row["count"] = missing_test_row["count"]*100
    fig, axes = plt.subplots(1,2, figsize=(18,6))
    sns.barplot( y =missing_train_row["count"] ,  x  = missing_train_row["no"],ax = axes[1] ,palette = "viridis")
    sns.barplot( y =missing_test_row["count"] ,  x  = missing_test_row["no"],ax = axes[0] ,palette = "viridis")
    axes[0].set_ylabel("Percentage of Null values")
    axes[1].set_ylabel("Percentage of Null values")
    axes[0].set_xlabel("TRAIN DATASET")
    axes[1].set_xlabel("TEST DATASET");


def plot_target(train: pd.DataFrame, target_col: str, objective: str):
    """
    Plots the target variable based on the objective.

    Args:
        train (pd.DataFrame): The training dataset.
        target_col (str): Name of the target column.
        objective (str): Objective of the analysis ('classification' or 'regression').

    Returns:
        None

    Raises:
        ValueError: If the objective is not 'classification' or 'regression'.
    """

    if objective == 'classification':
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_col, data=train)
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.title(f"Count Plot of {target_col}")
        plt.show()

    elif objective == 'regression':
        plt.figure(figsize=(8, 6))
        sns.histplot(x=target_col, data=train, kde=True)
        plt.xlabel(target_col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution Plot of {target_col}")
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.boxplot(x=target_col, data=train)
        plt.xlabel(target_col)
        plt.ylabel("Value")
        plt.title(f"Box Plot of {target_col}")
        plt.show()

    else:
        raise ValueError("Invalid objective. Must be 'classification' or 'regression'.")        




def plotNumericalDistributionOnTopOfEachOther(train: pd.DataFrame, test: pd.DataFrame, numerical_cols: list):
    """
    Plots numerical distribution for train and test data on top of each other, with flexibility for handling missing columns.

    Args:
        train (pd.DataFrame): The training data containing numerical columns.
        test (pd.DataFrame): The test data containing numerical columns.
        numerical_cols (list): List of numerical column names.

    Returns:
        None

    Raises:
        None

    """
    ncols = 5
    num_cols = len(numerical_cols)
    nrows = math.ceil(num_cols / ncols)  # Calculate the number of rows dynamically

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), facecolor='#EAEAF2')
    fig.tight_layout(pad=3.0)  # Adjust spacing between subplots

    if nrows == 1:
        axes = [axes]  # Wrap the axes object in a list to handle the single-row case

    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c  # Calculate the corresponding index for the numerical column
            if idx < num_cols:
                col = numerical_cols[idx]
                ax = axes[r][c]  # Access the subplot using two indices

                sns.histplot(x=train[col], ax=ax, color='#58D68D', label='Train data', fill=True, kde=True)

                if col in test.columns:  # Check if the column exists in the test data
                    sns.histplot(x=test[col], ax=ax, color='#DE3163', label='Test data', fill=True, kde=True)

                ax.legend()
                ax.set_ylabel('')
                ax.set_xlabel(col, fontsize=12)
                ax.tick_params(labelsize=10, width=0.5)
                ax.xaxis.offsetText.set_fontsize(8)
                ax.yaxis.offsetText.set_fontsize(8)
            else:
                fig.delaxes(axes[r][c])  # Remove unused subplot

    plt.show()


    def plotCategoricalDistributionOnTopOfEachOther(train: pd.DataFrame, test: pd.DataFrame, categorical_cols: list):
        """
        Plots categorical distribution for train and test data side by side, with flexibility for handling missing columns.

        Args:
            train (pd.DataFrame): The training data containing categorical columns.
            test (pd.DataFrame): The test data containing categorical columns.
            categorical_cols (list): List of categorical column names.

        Returns:
            None

        Raises:
            None

        """
    if len(categorical_cols) == 0:
        print("No Categorical features")
        return

    num_cols = len(categorical_cols)
    ncols = 3
    nrows = math.ceil(num_cols / ncols)  # Calculate the number of rows dynamically

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.reshape(-1, ncols)  # Reshape axes to be 2-dimensional

    fig.tight_layout(pad=3.0)  # Adjust spacing between subplots

    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c  # Calculate the corresponding index for the categorical column
            if idx < num_cols:
                col = categorical_cols[idx]
                ax = axes[r, c]

                if col in test.columns:  # Check if the column exists in the test data
                    unique_train = train[col].unique()
                    unique_test = test[col].unique()

                    if set(unique_test).issubset(set(unique_train)):
                        sns.countplot(data=train, x=col, ax=ax, palette="viridis", label='Train data')
                        sns.countplot(data=test, x=col, ax=ax, palette="magma", label='Test data')
                    else:
                        print("No similar categories in the test dataset for column:", col)
                        sns.countplot(data=train, x=col, ax=ax, palette="viridis", label='Train data')
                else:
                    sns.countplot(data=train, x=col, ax=ax, palette="viridis", label='Train data')

                ax.legend()
                ax.set_ylabel('')
                ax.set_xlabel(col, fontsize=12)
                ax.tick_params(labelsize=10, width=0.5)
                ax.xaxis.offsetText.set_fontsize(8)
                ax.yaxis.offsetText.set_fontsize(8)
            else:
                fig.delaxes(axes[r, c])  # Remove unused subplot

    plt.show()

def plotCategoricalDistributionSideBySide(dataframe:pd.DataFrame, categorical_columns:list, test_dataframe: pd.DataFrame =None):
    """
    Plots the distribution of categorical columns in the provided dataframe(s).

    Args:
        dataframe (pandas.DataFrame): The train dataset or the single dataset to plot.
        categorical_columns (list): List of categorical column names to plot.
        test_dataframe (pandas.DataFrame, optional): The test dataset to plot alongside the train dataset.

    Returns:
        None
    """

    num_plots = len(categorical_columns)
    num_datasets = 1 if test_dataframe is None else 2

    if num_datasets == 2:
        fig, axes = plt.subplots(nrows=num_plots, ncols=num_datasets, figsize=(10, num_plots * 5))
    else:
        fig, axes = plt.subplots(nrows=num_plots, ncols=num_datasets, figsize=(6, num_plots * 5))

    if num_plots == 1:
        axes = [axes]

    for i, column in enumerate(categorical_columns):
        if num_datasets == 2:
            ax_train = axes[i][0]
            ax_test = axes[i][1]
        else:
            ax_train = axes[i]

        if num_datasets == 2:
            dataframe[column].value_counts().plot(kind='bar', ax=ax_train, color='blue', alpha=0.7)
            ax_train.set_title(f"Train Dataset - {column} Distribution")
            ax_train.set_xlabel(column)
            ax_train.set_ylabel("Frequency")
            ax_train.tick_params(axis='x', rotation=45)  # Rotate x-labels

            if column in test_dataframe.columns:
                test_dataframe[column].value_counts().plot(kind='bar', ax=ax_test, color='orange', alpha=0.7)
                ax_test.set_title(f"Test Dataset - {column} Distribution")
                ax_test.set_xlabel(column)
                ax_test.set_ylabel("Frequency")
                ax_test.tick_params(axis='x', rotation=45)  # Rotate x-labels
            else:
                ax_test.axis('off')  # Hide the empty subplot if column is not present in test dataframe

            # Add a vertical black line between train and test plots
            ax_train.axvline(len(ax_train.patches), color='black', linestyle='dashed')
        else:
            dataframe[column].value_counts().plot(kind='bar', ax=ax_train, color='blue', alpha=0.7)
            ax_train.set_title(f"{column} Distribution")
            ax_train.set_xlabel(column)
            ax_train.set_ylabel("Frequency")
            ax_train.tick_params(axis='x', rotation=45)  # Rotate x-labels

    plt.tight_layout()
    plt.show()


    def plotNumericalDistributionSideBySide(dataframe, numerical_cols, test_dataframe=None):
        """
        Plots the scatter plot, box plot, and histogram plot of numerical columns in the provided dataframe(s).

        Args:
            dataframe (pandas.DataFrame): The train dataset or the single dataset to plot.
            numerical_cols (list): List of numerical column names to plot.
            test_dataframe (pandas.DataFrame, optional): The test dataset to plot alongside the train dataset.

        Returns:
            None
        """

    num_plots = len(numerical_cols)
    num_datasets = 1 if test_dataframe is None else 2

    fig, axes = plt.subplots(nrows=num_plots, ncols=(num_datasets * 3), figsize=(15, num_plots * 5))

    if num_plots == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, col in enumerate(numerical_cols):
        ax_train_scatter = axes[i, 0]
        ax_train_box = axes[i, 1]
        ax_train_hist = axes[i, 2]

        # Scatter plot - Train
        ax_train_scatter.scatter(range(dataframe.shape[0]), np.sort(dataframe[col].values), color='blue')
        ax_train_scatter.set_title("\n".join(textwrap.wrap(f"Train Dataset - {col} Scatter Plot", 20)))
        ax_train_scatter.set_ylabel(col)

        # Box plot - Train
        sns.boxplot(data=dataframe, x=col, ax=ax_train_box)
        ax_train_box.set_title("\n".join(textwrap.wrap(f"Train Dataset - {col} Box Plot", 20)))
        ax_train_box.set_xlabel(col)
        ax_train_box.set_ylabel("")

        # Histogram plot - Train
        sns.histplot(dataframe[col], ax=ax_train_hist, color='blue', kde=True)
        ax_train_hist.set_title("\n".join(textwrap.wrap(f"Train Dataset - {col} Histogram Plot", 20)))
        ax_train_hist.set_xlabel(col)
        ax_train_hist.set_ylabel("")

        if num_datasets == 2:
            ax_test_scatter = axes[i, 3]
            ax_test_box = axes[i, 4]
            ax_test_hist = axes[i, 5]

            if col in test_dataframe.columns:
                # Scatter plot - Test
                ax_test_scatter.scatter(range(test_dataframe.shape[0]), np.sort(test_dataframe[col].values), color='orange')
                ax_test_scatter.set_title("\n".join(textwrap.wrap(f"Test Dataset - {col} Scatter Plot", 20)))
                ax_test_scatter.set_ylabel(col)

                # Box plot - Test
                sns.boxplot(data=test_dataframe, x=col, ax=ax_test_box)
                ax_test_box.set_title("\n".join(textwrap.wrap(f"Test Dataset - {col} Box Plot", 20)))
                ax_test_box.set_xlabel(col)
                ax_test_box.set_ylabel("")

                # Histogram plot - Test
                sns.histplot(test_dataframe[col], ax=ax_test_hist, color='orange', kde=True)
                ax_test_hist.set_title("\n".join(textwrap.wrap(f"Test Dataset - {col} Histogram Plot", 20)))
                ax_test_hist.set_xlabel(col)
                ax_test_hist.set_ylabel("")

            else:
                ax_test_scatter.axis('off')
                ax_test_box.axis('off')
                ax_test_hist.axis('off')

        if num_datasets == 2:
            # Add boundary between train and test columns
            ax_train_hist.axvline(x=dataframe[col].max(), color='black', linestyle='--', linewidth=1)
            ax_test_hist.axvline(x=test_dataframe[col].max(), color='black', linestyle='--', linewidth=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplot spacing
    plt.show()

def plotCorrelationHeatMap(train: pd.DataFrame, numerical_cols: list):
    """
    Plots correlation heatmap for numerical columns in subsets to improve readability.

    Args:
        train (pd.DataFrame): The training data containing numerical columns.
        numerical_cols (list): List of numerical column names.

    Returns:
        None

    Raises:
        None

    """
    print("Total numerical columns:", len(numerical_cols))

    subset_length = 10  # Set the subset length for better visibility
    num_subsets = math.ceil(len(numerical_cols) / subset_length)

    for subset_idx in range(num_subsets):
        start_idx = subset_idx * subset_length
        end_idx = min(start_idx + subset_length, len(numerical_cols))
        subset_cols = numerical_cols[start_idx:end_idx]

        print("Subset:", subset_idx + 1)
        print(subset_cols)

        plt.rcParams["figure.figsize"] = (18, 12)
        dataplot = sns.heatmap(train[subset_cols].corr(), cmap="viridis", annot=True)
        plt.show()

def plot_relationship(train: pd.DataFrame, categorical_cols: list, numerical_cols: list, dependent_feature: str, objective: str):
    """
    Plots the relationship between independent features and the dependent feature based on the objective.

    Args:
        train (pd.DataFrame): The training dataset.
        categorical_cols (list): List of categorical column names.
        numerical_cols (list): List of numerical column names.
        dependent_feature (str): Name of the dependent feature.
        objective (str): Objective of the analysis ('classification' or 'regression').

    Returns:
        None

    Raises:
        ValueError: If the objective is not 'classification' or 'regression'.
    """

    if objective == 'classification':
        for feature in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=feature, hue=dependent_feature, data=train)
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.title(f"Count Plot of {feature} by {dependent_feature}")
            plt.show()

        for feature in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=train[feature], y=train[dependent_feature])
            plt.xlabel(feature)
            plt.ylabel(dependent_feature)
            plt.title(f"Box Plot of {dependent_feature} by {feature}")
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.violinplot(x=train[feature], y=train[dependent_feature])
            plt.xlabel(feature)
            plt.ylabel(dependent_feature)
            plt.title(f"Violin Plot of {dependent_feature} by {feature}")
            plt.show()

    elif objective == 'regression':
        for feature in numerical_cols:
            plt.figure(figsize=(10, 6))
            plt.scatter(train[feature], train[dependent_feature])
            plt.xlabel(feature)
            plt.ylabel(dependent_feature)
            plt.title(f"Scatter Plot of {dependent_feature} vs. {feature}")
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.lineplot(x=train[feature], y=train[dependent_feature])
            plt.xlabel(feature)
            plt.ylabel(dependent_feature)
            plt.title(f"Line Plot of {dependent_feature} vs. {feature}")
            plt.show()

    else:
        raise ValueError("Invalid objective. Must be 'classification' or 'regression'.")