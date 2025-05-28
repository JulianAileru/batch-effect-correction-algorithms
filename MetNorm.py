from .norm_class import normalization
import pandas as pd 
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import rankdata

class MetNorm(normalization):
    def __init__(self,data,metadata):
        super().__init__(data,metadata)
        self.sorted_signals = None
        self.scaler_y = StandardScaler()
        self.scaler_X = StandardScaler()
        self.QC_signal = None
        self.QC_idx = None
        self.sample_signal = None
        self.sample_signal_idx = None
        self.normed = None
    def _top_correlated(self,n=5,method="spearman"):
        corr = self.QC.T.corr(method=method,numeric_only=True) #spearman correlation
        unstack = corr.unstack() #long format of matrix format [Sig1,Sig2,correlation_value]
        signal_dict = {}
        for i,j in unstack.items(): #loop through series 
            lst = []
            sig1,sig2 = i
            if sig1 != sig2: #omit self-correlation
                if sig1 not in signal_dict: #store each signal in dict, create list and add (signal,correlation_value)
                    signal_dict[sig1] = lst
                    signal_dict.get(sig1).append((sig2,j))
                else:
                    signal_dict[sig1].append(((sig2,j)))
        self.sorted_signals = {i: [item[0] for item in sorted(j, key=lambda x: x[1], reverse=True)][:n] for i, j in signal_dict.items()} #sort items and store signal ids only, keep top n signals
        # return sorted_signals
    def _top_correlated(self,n=5,method='spearman'):
        QC = self.QC.T.copy()
        signals = QC.columns.tolist()
        QC_ranked = np.apply_along_axis(rankdata,axis=0,arr=QC)
        spearman_corr = np.corrcoef(QC_ranked, rowvar=False)
        signal_dict = {signal:None for signal in signals}
        for idx,signal in enumerate(signals):
            df = pd.Series(spearman_corr[:,idx],name=signal,index=signals)
            df = df.sort_values(ascending=False)
            df.drop(index=df.name,inplace=True)
            signal_dict[signal] = df.index.tolist()[:n]
        self.sorted_signals = signal_dict
    def _fit(self,signal,corr):
        regr = SVR(C=1, epsilon=0.1, kernel='rbf', tol=0.001, gamma='auto', cache_size=40) 
        #Get and reshape the target signal
        QC_signal = self.QC.loc[signal]
        self.QC_idx = QC_signal.index
        QC_signal = QC_signal.to_numpy().reshape(-1,1)
        self.QC_signal = QC_signal
        QC_signal_scaled = self.scaler_y.fit_transform(QC_signal)

        sample_signal = self.sample.loc[signal]
        self.sample_signal_idx = sample_signal.index
        sample_signal = sample_signal.to_numpy().reshape(-1,1)
        self.sample_signal = sample_signal

        #Get correlated features and scale them 
        X_train = self.QC.loc[corr].to_numpy().T
        X_test = self.sample.loc[corr].to_numpy().T
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        model = regr.fit(X_train_scaled,QC_signal_scaled.ravel())
        return model,X_train_scaled,X_test_scaled
    def _predict(self,model,X_train_scaled,X_test_scaled):
        #Get Fitted Values and Predictions on X_test_scaled
        QC_pred = model.predict(X_train_scaled).reshape(-1,1)
        sample_pred = model.predict(X_test_scaled).reshape(-1,1)
        return QC_pred,sample_pred
    def _normalize_signals(self,QC_pred,sample_pred):
        #Convert predictions back to original scale
        QC_pred_original = self.scaler_y.inverse_transform(QC_pred)
        sample_pred_original = self.scaler_y.inverse_transform(sample_pred.reshape(-1,1))
        #Normalize_signals
        QC_norm = self.QC_signal / QC_pred_original
        sample_norm = self.sample_signal / sample_pred_original
        return QC_norm,sample_norm
    def fit_transform(self):
        #compute sorted_signals dictionary
        #Iterate through dict of key:signal, value:correlated signals
        #call helper functions
        #concatenate into 1 dataframe
        qc_list = []
        sample_list = []
        self._top_correlated(method='spearman')
        for sig,cor in self.sorted_signals.items():
            model,X_train_scaled,X_test_scaled = self._fit(signal=sig,corr=cor)
            QC_pred,sample_pred = self._predict(model,X_train_scaled,X_test_scaled)
            QC_norm,sample_norm = self._normalize_signals(QC_pred,sample_pred)
            qc_list.append(pd.Series(QC_norm.flatten(), index=self.QC_idx, name=sig))
            sample_list.append(pd.Series(sample_norm.flatten(), index=self.sample_signal_idx, name=sig))
        self.QC_normed = pd.concat(qc_list,axis=1)
        self.sample_normed = pd.concat(sample_list,axis=1)
        self.normed = pd.concat([self.QC_normed,self.sample_normed],axis=0)
        return self.normed

"""
Testing: 

    D = pd.read_csv("~/MetNormalizer/2-peak_area_after_filling_missing_values.csv")
    M = pd.read_csv("~/MetNormalizer/sample_metadata_all_batches.csv")
    x = MetNorm(D,M)
    x.fit_transform()
    MetNorm._pca_plot(x.normed,x.metadata,hue="sample_type")
    
"""