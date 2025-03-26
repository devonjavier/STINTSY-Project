import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from src.models.LogisticRegression import LogisticRegression, LFSDataset

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import ParameterSampler

import random

def set_seeds(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    
def prepare_data_loaders(data_dict, batch_size, missing_value):
    train_dataset = LFSDataset(data_dict['X_train'], data_dict['y_train'], missing_value)
    test_dataset = LFSDataset(data_dict['X_test'], data_dict['y_test'], missing_value)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def initialize_model(input_dim, learning_rate, scheduler_step_size, scheduler_gamma, weight_decay, optimizer_name, seed):

    model = LogisticRegression(input_dim, seed)
    criterion = torch.nn.BCELoss()
    
    if optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose 'sgd', 'adam', or 'rmsprop'.")
    
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    return model, criterion, optimizer, scheduler

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in train_loader:
        features = batch['features']
        mask = batch['mask']
        labels = batch['labels'].view(-1, 1)
        
        outputs = model(features, mask)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_predictions = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            mask = batch['mask']
            labels = batch['labels'].view(-1, 1)
            
            outputs = model(features, mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.view(-1).tolist())
            all_predictions.extend(predicted.view(-1).tolist())
    
    return total_loss / len(test_loader), correct / total, all_labels, all_predictions

def log_metrics(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits = 8))
    return cm

def train_model(data_dict, learning_rate=0.01, batch_size=128, num_epochs=50, 
                scheduler_step_size=5, scheduler_gamma=0.5, missing_value=-1,
                convergence_threshold=1e-4, patience=3, weight_decay=0,
                optimizer='sgd',
                seed = 45):  
    
    set_seeds(seed)
    train_loader, test_loader = prepare_data_loaders(data_dict, batch_size, missing_value)
    input_dim = data_dict['X_train'].shape[1]
    
    
    model, criterion, optimizer, scheduler = initialize_model(
        input_dim, 
        learning_rate, 
        scheduler_step_size, 
        scheduler_gamma, 
        weight_decay=weight_decay,
        optimizer_name=optimizer,
        seed=seed
    )
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
    }
    
    best_test_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_accuracy, all_y_test, all_predictions = evaluate_model(model, test_loader, criterion)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_accuracy'].append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, '
              f'Train Acc: {train_accuracy:.4f}, '
              f'Test Acc: {test_accuracy:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        
        if test_loss < best_test_loss - convergence_threshold:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
    
    cm = log_metrics(all_y_test, all_predictions)
    
    return {
        'model': model,
        'history': history,
        'feature_names': data_dict['feature_names'],
        'confusion_matrix': cm
    }

def hyperparameter_random_search(param_distributions, data_dict, n_iter_search=50):
    
    random_params = list(ParameterSampler(
        param_distributions, 
        n_iter=n_iter_search, 
        random_state=42
    ))
    
    results = []
    
    
    for params in random_params:
        print("\nTesting configuration:")
        print(params)
        
        try:
            result_dict = train_model(
                data_dict, 
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                optimizer=params['optimizer'],
                weight_decay=params['weight_decay'],
                num_epochs=params['num_epochs'],
                scheduler_step_size=params['scheduler_step_size'],
                scheduler_gamma=params['scheduler_gamma'],
                patience=params['patience']
            )
            
            
            final_test_accuracy = result_dict['history']['test_accuracy'][-1]
            final_test_loss = result_dict['history']['test_loss'][-1]
            
            
            results.append({
                'params': params,
                'test_accuracy': final_test_accuracy,
                'test_loss': final_test_loss,
                'full_result': result_dict
            })
            
        except Exception as e:
            print(f"Error in configuration: {e}")
    
    
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    
    summary_df = pd.DataFrame([
        {
            'Learning Rate': r['params']['learning_rate'],
            'Batch Size': r['params']['batch_size'],
            'Optimizer': r['params']['optimizer'],
            'Test Accuracy': r['test_accuracy'],
            'Test Loss': r['test_loss']
        } for r in results
    ])
    
    
    plt.figure(figsize=(12, 6))
    plt.scatter(summary_df['Learning Rate'], summary_df['Test Accuracy'], 
                c=summary_df['Batch Size'], cmap='viridis', alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Test Accuracy')
    plt.title('Hyperparameter Random Search Results')
    plt.colorbar(label='Batch Size')
    plt.tight_layout()
    plt.show()
    
    
    print("\nTop 5 Configurations:")
    print(summary_df.head())
    
    
    best_result = results[0]
    print("\nBest Configuration:")
    print(f"Best Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"Best Params: {best_result['params']}")
    
    return {
        'best_params': best_result['params'],
        'results': results,
        'summary_df': summary_df
    }