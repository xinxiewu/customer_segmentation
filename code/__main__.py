"""
__maim__.py contains the workflow to run all sub-programs
"""
from util import *
from models import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def main(fileurl=None, output=None, continuous=None, nominal=None, epoch=None):
    """
    Step 1: Data Preparation & EDA
    """
    # 1.1 Download dataset from Github & read as DataFrame
    df = download_file(fileurl)
    # df = pd.read_csv('wholesale_customers_data.csv')
    print(f"Loaded dataset with {df.shape}\n")
    # 1.2 EDA
    # 1.2.1 Attributes Distributions (outliers) & Missing Values
    print(f"EDA Part")
    miss_yn, miss = missing_detect(df=df,cols=df.columns)
    fname = 'basic_statistic.csv'
    df.describe().T.to_csv(os.path.join(output, fname))
    print(f"Basic statistic saved as {fname}")
    fname = variable_dist(df=df, cols=df.columns)
    print(f"Variables' original distribution saved as {fname}")
    # 1.2.2 Correlation Heatmap
    fname = 'corr_heat_map.png'
    plt.subplots(figsize=(8,8))
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True)
    plt.savefig(os.path.join(output, fname))
    print(f"Variables' correlation heatmap saved as {fname}")
    # 1.2.2 Feature Engineering
    """
        Continuous: Right-skew -> Log -> Standardization
        Nominal: On-hot encoding
    """
    df_1 = feature_engineering(df=df, continuous=True, cols=continuous)
    df_2 = feature_engineering(df=df_1, cols=nominal)
    # df_1.to_csv(os.path.join(output, 'data1.csv'), index=False)
    # df_2.to_csv(os.path.join(output, 'data2.csv'), index=False)
    print(f'Dataset is READY!\n')

    """
    Step 2: Clustering Algorithms
    """
    # 2.1 Centroid-based: K-means
    fname = kmeans(df_list=[df,df_1,df_2], cluster_high=15)
    print(f"K-means results saved as {fname}\n")
    print(f"Project Ends!")
    return

if __name__ == '__main__':
    """
    Step 1: Clean output folder
    """
    delete_files()
    """
    Step 2: Call the main program
    """
    main(fileurl = 'https://raw.githubusercontent.com/xinxiewu/datasets/main/customer_segmentation/wholesale_customers_data.csv',
         output = r'../public/output',
         continuous = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'],
         nominal = ['Channel', 'Region'],
         epoch = 100
         )