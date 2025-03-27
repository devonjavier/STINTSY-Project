import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def prepare_data_kfold(lfs_data, 
                        target_col='PUFC11_WORK',
                        feature_cols=None,
                        categorical_cols=None,
                        n_splits=5,
                        missing_value=-1,
                        seed=45):
    print("Preparing data for k-fold cross-validation...")
    
    
    if feature_cols is None:
        feature_cols = [col for col in lfs_data.columns if col != target_col]
    if categorical_cols is None:
        categorical_cols = []
    
    
    filtered_data = lfs_data[lfs_data[target_col] != missing_value][feature_cols + [target_col]]
    
    X = filtered_data[feature_cols]
    y = filtered_data[target_col]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_data = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        print(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
        print(f"Testing on {len(X_test)} samples")
        print(f"Features: {feature_cols}")
        
        
        X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, dummy_na=True)
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, dummy_na=True)
        
        
        all_columns = list(set(X_train_encoded.columns) | set(X_test_encoded.columns))
        X_train_encoded = X_train_encoded.reindex(columns=all_columns, fill_value=0)
        X_test_encoded = X_test_encoded.reindex(columns=all_columns, fill_value=0)
        
        
        numerical_cols = [col for col in all_columns 
                          if col not in X_train_encoded.columns[X_train_encoded.dtypes == 'uint8']]
        
        
        if numerical_cols:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_encoded[numerical_cols]),
                columns=numerical_cols,
                index=X_train_encoded.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test_encoded[numerical_cols]),
                columns=numerical_cols,
                index=X_test_encoded.index
            )
            
            X_train_encoded[numerical_cols] = X_train_scaled
            X_test_encoded[numerical_cols] = X_test_scaled
        else:
            scaler = None
        
        folds_data.append({
            'X_train': X_train_encoded,
            'X_test': X_test_encoded,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train_encoded.columns,
            'scaler': scaler
        })
    
    return folds_data