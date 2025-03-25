import numpy as np
import torch
from torch.utils.data import Dataset


class LFSDataset(Dataset):
    def __init__(self, features, labels, missing_value=-1):
        self.features = features.values.astype(np.float32)
        
        self.labels = (labels.values == 2).astype(np.float32)
        self.missing_value = missing_value
        
        self.mask = (self.features != missing_value).astype(np.float32)
        self.features = np.where(self.features == missing_value, 0, self.features)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'mask': self.mask[idx],
            'labels': self.labels[idx]
        }


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        
    def forward(self, features, mask):
        masked_features = features * mask
        output = self.linear(masked_features)
        return torch.sigmoid(output)


def analyze_model(result_dict):
    model = result_dict['model']
    feature_names = result_dict['feature_names']
    
    weights = model.linear.weight.data.numpy().flatten()
    bias = model.linear.bias.data.numpy()[0]
    
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': weights
    })
    
    coefficients = coefficients.reindex(coefficients['Coefficient'].abs().sort_values(ascending=False).index)
    
    print("\nLogistic Regression Coefficients:")
    print(f"Bias (Intercept): {bias:.4f}")
    print(coefficients)

def predict_employment_status(lfs_data, result_dict, target_normalization=None, missing_value=-1):
    model = result_dict['model']
    feature_names = result_dict['feature_names']
    
    X = lfs_data[feature_names]
    
    mask = (X != missing_value).astype(np.float32)
    X = X.replace(missing_value, 0).astype(np.float32)
    
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X.values), torch.tensor(mask.values))
        binary_predictions = (predictions >= 0.5).float().numpy().flatten()
     
    if target_normalization is not None and 'original_values' in target_normalization:
        original_values = target_normalization['original_values']
        if len(original_values) == 2:
            value_map = {0: min(original_values), 1: max(original_values)}
            return pd.Series([value_map[int(p)] for p in binary_predictions], index=X.index)
    
    return pd.Series(binary_predictions, index=X.index)



def predict_employment_status(lfs_data, result_dict, target_normalization=None, missing_value=-1):
    model = result_dict['model']
    feature_names = result_dict['feature_names']
    
    X = lfs_data[feature_names]
    
    mask = (X != missing_value).astype(np.float32)
    X = X.replace(missing_value, 0).astype(np.float32)
    
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X.values), torch.tensor(mask.values))
        binary_predictions = (predictions >= 0.5).float().numpy().flatten()
     
    if target_normalization is not None and 'original_values' in target_normalization:
        original_values = target_normalization['original_values']
        if len(original_values) == 2:
            value_map = {0: min(original_values), 1: max(original_values)}
            return pd.Series([value_map[int(p)] for p in binary_predictions], index=X.index)
    
    return pd.Series(binary_predictions, index=X.index)