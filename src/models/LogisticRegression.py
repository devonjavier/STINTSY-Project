import numpy as np
import torch
from torch.utils.data import Dataset


class LFSDataset(Dataset):
    def __init__(self, features, labels, missing_value=-1):
        self.features = features.values.astype(np.float32)
        # Convert labels to binary (1 for did work this past week, 0 for did not work)
        self.labels = (labels.values == 2).astype(np.float32)
        self.missing_value = missing_value
        # Create a mask to indicate missing values
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
    def __init__(self, input_dim, seed=45):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(input_dim, 1)
        # Initialize weights and bias
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
    """
    Perform predictions using an aggregate approach across all folds
    
    Parameters:
    - lfs_data: Original dataset
    - result_dict: Results from train_model() with multiple folds
    - target_normalization: Optional normalization for target variable
    - missing_value: Value representing missing data
    
    Returns:
    - Aggregated predictions for the entire dataset
    """
    from torch.utils.data import DataLoader
    from src.models.LogisticRegression import LFSDataset
    import torch
    import numpy as np
    import pandas as pd

    # Extract feature names from the first fold (assuming consistent across folds)
    feature_names = result_dict['fold_results'][0]['feature_names']
    
    # Prepare the input data
    X_predict = lfs_data[feature_names]
    
    # Collect predictions from all folds
    all_fold_predictions = []
    
    for fold in result_dict['fold_results']:
        # Retrieve the trained model for this fold
        model = fold.get('model')  # You'll need to modify train_model to save the model
        
        if model is None:
            print(f"Warning: No model found for a fold. Skipping.")
            continue
        
        # Prepare prediction dataset
        predict_dataset = LFSDataset(X_predict, 
                                     torch.zeros(len(X_predict)), 
                                     missing_value)
        predict_loader = DataLoader(predict_dataset, batch_size=128)
        
        # Collect predictions for this fold
        fold_predictions = []
        model.eval()
        with torch.no_grad():
            for batch in predict_loader:
                features = batch['features']
                mask = batch['mask']
                outputs = model(features, mask)
                fold_predictions.extend(outputs.numpy())
        
        all_fold_predictions.append(fold_predictions)
    
    # Aggregate predictions (simple averaging)
    aggregate_predictions = np.mean(all_fold_predictions, axis=0)
    
    # Convert to binary predictions
    binary_predictions = (aggregate_predictions >= 0.5).astype(int)
    
    # Optional: Add predictions back to original dataframe
    lfs_data['predicted_employment_status'] = binary_predictions
    
    # Detailed results
    results = {
        'raw_probabilities': aggregate_predictions,
        'binary_predictions': binary_predictions,
        'prediction_summary': {
            'total_predictions': len(binary_predictions),
            'positive_predictions': np.sum(binary_predictions),
            'negative_predictions': len(binary_predictions) - np.sum(binary_predictions),
            'positive_rate': np.mean(binary_predictions)
        }
    }
    
    return results