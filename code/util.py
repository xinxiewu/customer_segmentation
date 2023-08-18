"""
util.py contains custom functions:
    1. download_file: Download the .csv file from the given link and read as dataframe
    2. delete_files: Delete files in the given folder, except README.md
    3. missing_detect: Detect if any missing value for each variable
    4. variable_dist: Generate plots of variables' distribution and save
    5. feature_engineering: Convert attributes as needed
"""
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import category_encoders as ce

# download_file(url, output)
def download_file(url=None, output=r'../public/output'):
    """ Download the .csv file from the given link and read as dataframe

    Args: 
        url: str
        output: path to store downloaded files
    
    Returns:
        DataFrame
    """
    local_filename = os.path.join(output, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return pd.read_csv(local_filename)

# delete_files(path, keep)
def delete_files(path=r'../public/output', keep=['README.md']):
    """ Delete files in the given folder path, except README.md

    Args:
        path: path, starting with r''
        keep: files to keep, default value as README.md

    Returns:
        nothing to return
    """
    for fname in os.listdir(path):
        if fname not in (keep):
            os.remove(os.path.join(path, fname))
    return

# missing_detect()
def missing_detect(df=None, cols=None):
    """ Detect if any missing value for each variable

    Args:
        df: input dataframe
        cols: columns from the input dataframe, df.columns

    Returns:
        return True if no missing values; otherwise, return False and variables with missing value
    """
    cols, res = list(cols), []
    for col in cols:
        if getattr(df, col).isna().sum() != 0:
            res.append(col)
    if len(res) == 0:
        print(f"No missing values!")
        return True, res
    else:
        print(f"Missing values in {res}")
        return False, res

# variable_dist():
def variable_dist(df=None, cols=None, nominal=['Channel', 'Region'], agg='Fresh', fname='variable_dist.png',
                  output=r'../public/output', subplots=[3,3], figsize=(40,30)):
    """ Generate plots of variables' distribution and save

    Args:
        df: input dataframe
        cols: columns from the input dataframe, df.columns

    Returns:
        fname
    """
    sns.set()
    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    ax, i, cols = axes.flatten(), 0, list(cols)
    for col in cols:
        if col in nominal:
            res = df.groupby(by=[col]).agg({agg: 'count'})
            res['%'] = res.apply(lambda x: 100*x/x.sum())
            res = res.reset_index()
            s = ''
            for j in res[col]:
                if len(s) > 0:
                    s += ', '
                s += f"{j}: {format(res[res[col]==j]['%'][j-1], '.2f')}%"
            temp = sns.countplot(data=df, x=col, ax=ax[i])
            temp.set(xlabel=s, ylabel='Count of {col}', title=f"Countplot of {col}")
        else:
            skew, kurt = format(getattr(df, col).skew(), '.2f'), format(getattr(df, col).kurt(), '.2f')
            if float(skew) > 2:
                skewness = 'Right-skewed'
            elif float(skew) <=2 and float(skew) >= -2:
                skewness = 'No-skewed'
            else:
                skewness = 'Left-skewed'
            if float(kurt) > 3:
                kurtosis = 'High-peak'
            elif float(kurt) <=3 and float(kurt) >= -3:
                kurtosis = 'Approx Normal Bell'
            else:
                kurtosis = 'Low-peak'
            temp = sns.histplot(getattr(df, col), kde=True, color='purple', ax=ax[i])
            temp.set(xlabel=f"Skew {skew}, Kurt {kurt}", ylabel='Density', title=f"Histogram of {col}: {skewness} with {kurtosis}")
        i += 1
    temp = sns.boxplot(data=df, ax=ax[i])
    temp.set(xlabel=f"Attributes", ylabel='Box', title=f"Boxplot of Attributes")
    plt.tight_layout()
    plt.savefig(os.path.join(output, fname))
    return fname

# feature_engineering():
def feature_engineering(df=None, continuous=False, cols=None, fname='continuous_conversion.png',
                        output=r'../public/output', subplots=[6,3], figsize=(40,30)):
    """ Convert attributes as needed and save plots

    Args:
        df: input dataframe
        cols: columns from the input dataframe

    Returns:
        Nothing to return
    """
    res_df = df.copy()
    if continuous == True:
        sns.set()
        fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
        ax, i = axes.flatten(), 0

        for col in cols:
            # Original Distribution
            temp = sns.histplot(getattr(res_df, col), kde=True, color='purple', ax=ax[i])
            temp.set(xlabel=col, ylabel='Density', title=f"Originals of {col}")
            i += 1
            # Log Distribution
            res_df[col] = np.log(res_df[col])
            temp = sns.histplot(getattr(res_df, col), kde=True, color='purple', ax=ax[i])
            temp.set(xlabel=col, ylabel='Density', title=f"Logs of {col}")
            i += 1
            # Standardization Distribuion
            scaler = preprocessing.StandardScaler()
            res_df[col] = scaler.fit_transform(pd.DataFrame(res_df[col]))
            temp = sns.histplot(getattr(res_df, col), kde=True, color='purple', ax=ax[i])
            temp.set(xlabel=col, ylabel='Density', title=f"Standardization of {col}")
            i += 1

        plt.tight_layout()
        plt.savefig(os.path.join(output, fname))
        print(f"Attributes standardization saved as {fname}")
    elif continuous == False:
        for col in cols:
            ohc = ce.OneHotEncoder(cols=col, return_df=True, use_cat_names=True)
            res_df = (ohc.fit_transform(res_df))
        res_df = res_df.rename(columns={'Channel_1.0':'Channel_1', 'Channel_2.0':'Channel_2', 'Region_1.0':'Region_1', 'Region_2.0':'Region_2', 'Region_3.0':'Region_3'})
        print(f"Nominal Attributes encoded with one-hot method")

    return res_df