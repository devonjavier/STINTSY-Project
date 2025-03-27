import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np

def prepare_data_kfold(lfs_data, target_col='PUFC11_WORK',
                        feature_cols=None,
                        categorical_cols=None,
                        n_splits=5, 
                        missing_value=-1,
                        seed=45):

    print("Preparing data for k-fold cross-validation...")
    # If no feature columns are provided, use all columns except the target column
    filtered_data = lfs_data[lfs_data[target_col] != missing_value][feature_cols + [target_col]]

    X = filtered_data[feature_cols]
    y = filtered_data[target_col]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_data = []
    # Iterate over the KFold splits
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
        print(f"Testing on {len(X_test)} samples")
        print(f"Features: {feature_cols}")

        X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, dummy_na=True)
        X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, dummy_na=True)
        # Make sure that the columns in the training and testing datasets are the same
        train_cols = X_train_encoded.columns
        test_cols = X_test_encoded.columns
        missing_cols_train = set(train_cols) - set(test_cols)
        missing_cols_test = set(test_cols) - set(train_cols)
        # Add missing columns to the testing and training datasets
        for c in missing_cols_train:
            X_test_encoded[c] = 0
        for c in missing_cols_test:
            X_train_encoded[c] = 0

        X_train_encoded = X_train_encoded[train_cols]
        X_test_encoded = X_test_encoded[train_cols]
        # Scale numerical columns
        numerical_cols = [col for col in X_train_encoded.columns if col not in categorical_cols]
        scaler = None
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

        folds_data.append({
            'X_train': X_train_encoded,
            'X_test': X_test_encoded,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train_encoded.columns,
            'scaler': scaler
        })

    return folds_data