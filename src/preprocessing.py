import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(lfs_data, target_col='PUFC11_WORK', feature_cols=None, test_size=0.2, missing_value=-1, seed=45):
    print("Preparing data...")  

    filtered_data = lfs_data[lfs_data[target_col] != missing_value][feature_cols + [target_col]]

    X = filtered_data[feature_cols]
    y = filtered_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    print(f"Training on {len('X_train')} samples with {len('feature_cols')} features")
    print(f"Features: {feature_cols}")
    X_train_filled = X_train.replace(missing_value, 0)
    X_test_filled = X_test.replace(missing_value, 0)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_filled),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_filled),
        columns=X_test.columns,
        index=X_test.index
    )
    
    X_train_scaled = X_train_scaled.where(X_train != missing_value, missing_value)
    X_test_scaled = X_test_scaled.where(X_test != missing_value, missing_value)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler
    }