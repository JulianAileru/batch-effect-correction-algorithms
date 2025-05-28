import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt 

#Subsequent QC-based normalization classes will inherit this class for PCA and data shape management 
class Normalization:
    def __init__(self,data,metadata):
        self.data = data.copy()
        self.metadata = metadata.copy()
        if self.data.shape[0] < self.data.shape[1]:
            print("data shape format should be: (n_samples x n_signals), transposing dataset...")
            self.data = self.data.T
        self.data.drop(columns=["position","mz","rt"],inplace=True)
        self.data.set_index("name",inplace=True)
        self.metadata.set_index("sample_name",inplace=True)
        df = pd.merge(self.data.T,self.metadata[["sample_type"]],right_index=True,left_index=True)
        self.QC = df[df["sample_type"] == "sp"].copy()
        self.sample = df[df["sample_type"] != "sp"].copy()
        self.QC.drop(columns="sample_type",inplace=True)
        self.sample.drop(columns="sample_type",inplace=True)
        self.QC = self.QC.T
        self.sample = self.sample.T
    @staticmethod
    def _pca_plot(D,M,hue,drop_blanks=True):
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        if D.shape[0] > D.shape[1]:
            print("transposing data to merge with metadata (sample-wise)...")
            D = D.T
        df = pd.merge(D,M[["batch","sample_type"]],left_index=True,right_index=True)
        if drop_blanks:
            df = df[df["sample_type"] != "blank"]
            batch_index = df["batch"]
            sample_type = df["sample_type"]
        else:
            batch_index = df["batch"]
            sample_type = df['sample_type']
        df.drop(columns=["batch","sample_type"],inplace=True)
        pca_df = pd.DataFrame(pca.fit_transform(scaler.fit_transform(df)),columns=["PC1","PC2"],index=df.index)
        pca_df['batch'] = batch_index
        pca_df["sample_type"] = sample_type
        sns.scatterplot(pca_df,x="PC1",y="PC2",hue=hue)
        plt.show()
        return pca_df
