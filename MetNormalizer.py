import pandas as pd 
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.model_selection import GridSearchCV

class MetNorm:
    def __init__(self,data,metadata):
        self.data = data.copy() 
        self.metadata = metadata.copy()
        self.model = SVR(C=1, epsilon=0.001, kernel='rbf', tol=0.001, gamma='auto', cache_size=40)
        self.QC = self.data[self.data.index.str.contains("_SP_")]
        self.QC_idx = self.QC.index
        self.sample = self.data[~self.data.index.str.contains("_SP_")]
        self.sample_idx = self.sample.index
        self.sorted_signals = None
        self.scaler_y = None
        self.scaler_X = None
        self.QC_signal = None
        self.sample_signal = None
        self.sample_signal_idx = None
        self.normed = None
        self.param_grid = {'kernel': ['rbf', 'linear','poly'],'C': [0.1, 1, 10, 100],'gamma': ['scale','auto', 0.01, 0.1, 1],'epsilon': [0.01, 0.1, 0.5]}
        
    def _top_correlated(self,n=5,method='spearman'):
        QC = self.QC.copy()
        signals = QC.columns.tolist()
        QC_ranked = np.apply_along_axis(rankdata,axis=0,arr=QC)
        spearman_corr = np.corrcoef(QC_ranked, rowvar=False)
        signal_dict = {signal:None for signal in signals}
        for idx,signal in enumerate(signals):
            df = pd.Series(spearman_corr[:,idx],name=signal,index=signals)
            df = df.sort_values(ascending=False,key=abs)
            df.drop(index=df.name,inplace=True)
            signal_dict[signal] = df.index.tolist()[:n]
        self.sorted_signals = signal_dict
    def _fit(self,signal,corr,CV=5):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_train = self.QC.loc[:,corr].to_numpy()
        X_train = self.scaler_X.fit_transform(X_train)

        y_train = self.QC.loc[:,signal].to_numpy()
        y_train = y_train.reshape(-1,1)
        y_train = self.scaler_y.fit_transform(y_train).ravel()
        if CV:
            svr = SVR()
            grid_search = GridSearchCV(svr,self.param_grid,cv=CV,scoring = 'r2',n_jobs=-1)
            grid_search.fit(X_train,y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_

        X_test = self.sample.loc[:,corr]
        X_test = self.scaler_X.transform(X_test)
        self.model.fit(X_train,y_train)
        self.X_test = X_test
        self.X_train = X_train
    def _predict(self):
        QC_pred = self.model.predict(self.X_train)
        sample_pred = self.model.predict(self.X_test)
        self.QC_pred = self.scaler_y.inverse_transform(QC_pred.reshape(-1,1))
        self.sample_pred = self.scaler_y.inverse_transform(sample_pred.reshape(-1,1))
    def _normalize_signals(self,signal):
        QC_norm = (self.QC.loc[:,signal] / self.QC_pred.ravel()).to_numpy()
        sample_norm = (self.sample.loc[:,signal] / self.sample_pred.ravel()).to_numpy()
        return QC_norm,sample_norm
    def fit_transform(self):
        qc_list = []
        sample_list = []
        self._top_correlated(method='spearman')
        for sig,cor in self.sorted_signals.items():
            self._fit(sig,cor)
            self._predict()
            QC_norm,sample_norm = self._normalize_signals(sig)
            qc_list.append(pd.Series(QC_norm.flatten(), index=self.QC_idx, name=sig))
            sample_list.append(pd.Series(sample_norm.flatten(), index=self.sample_idx, name=sig))
        self.QC_normed = pd.concat(qc_list,axis=1)
        self.sample_normed = pd.concat(sample_list,axis=1)
        self.normed = pd.concat([self.QC_normed,self.sample_normed],axis=0)
        return self.normed