import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(lfs_data, target_col='PUFC11_WORK',
                 feature_cols=None,
                 categorical_cols=None,
                 test_size=0.2,
                 missing_value=-1,
                 seed=45):
    print("Preparing data...")  

    filtered_data = lfs_data[lfs_data[target_col] != missing_value][feature_cols + [target_col]]

    X = filtered_data[feature_cols]
    y = filtered_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    print(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
    print(f"Features: {feature_cols}")

    
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, dummy_na=True) 
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, dummy_na=True)

    
    train_cols = X_train_encoded.columns
    test_cols = X_test_encoded.columns
    missing_cols_train = set(train_cols) - set(test_cols)
    missing_cols_test = set(test_cols) - set(train_cols)

    for c in missing_cols_train:
        X_test_encoded[c] = 0
    for c in missing_cols_test:
        X_train_encoded[c] = 0

    X_train_encoded = X_train_encoded[train_cols]
    X_test_encoded = X_test_encoded[train_cols]

    
    numerical_cols = [col for col in X_train_encoded.columns if col not in categorical_cols] 
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

    return {
        'X_train': X_train_encoded,
        'X_test': X_test_encoded,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X_train_encoded.columns,
        'scaler': scaler if numerical_cols else None
    }