import numpy as np
import pandas as pd 
import patsy
import statsmodels.api as sm 
def batchEffectCorrection(D, M, method='ls'):
    """
    Limma Batch Effect Correction 

    Arguments
    ---------
    D: pd.DataFrame
        - Data Matrix of shape (n_signals, n_samples)
    M: pd.DataFrame
        - MetaData Matrix, can have other covariate information, needs to have batch variable defined
    method: str {ls ...}
        - Apply OLS to each signal
        - Could add other options such as WLS, GLS
        
    Returns
    -------
    pd.DataFrame
        - Batch-corrected data with same shape as input D (n_signals, n_samples)
    """
    if D.shape[0] < D.shape[1]:
        raise ValueError("data matrix is expected to be shape (n_signals, n_samples)")
    
    if method == "ls":
        #MAKE SURE data and batch labels have the same ordering
        M = M.loc[D.columns]

        # Initialize design matrix with deviation encoding of categorical variables
        design = patsy.dmatrix("1 + C(batch, Sum)", data=M)
        n_signals, n_samples = D.shape
        models = []
        n_batches = len(pd.Categorical(M["batch"]).categories)
        
        # Apply signal-wise OLS using statsmodels function
        for i in range(n_signals):
            model = sm.OLS(D.iloc[i,:], design)
            results = model.fit(method='qr')
            models.append(results)
            
        # Extract parameters (intercept and batch effect)
        betas = np.array([model.params for model in models])
        
        # Select batch effect parameter(s)
        batch_params = betas[:, 1:n_batches]
        
        # Use all batch indicators from the design matrix
        batch_design = np.asarray(design[:, 1:n_batches])
        
        # Subtract batch effect contribution from data
        batch_effect = batch_design @ batch_params.T
        adjusted_D = D - batch_effect.T  # Keep original shape
        
        return adjusted_D.T  # Shape (n_samples, n_signals)
    else:
        raise ValueError(f"Method '{method}' not implemented")

#Testing
"""
D = pd.read_csv("log2_transformed_metadata-attached_for_limma.csv")
D = D.rename(columns={"Unnamed: 0":"sample"})
D.set_index("sample",inplace=True)
M = D[["sample_type","batch","injection_order"]]
D.drop(columns=["sample_type","batch","injection_order"],inplace=True)
D = D.T

"""
