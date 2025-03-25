import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, classification_report
from src.models.LogisticRegression import LogisticRegression, NeuralNetwork, LFSDataset
import numpy as np
import random

def set_seeds(seed=45):
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

def initialize_model(input_dim, learning_rate, scheduler_step_size, scheduler_gamma, weight_decay, model, optimizer_name='sgd,'):

    if model == 'lr':
        model = LogisticRegression(input_dim)
    elif model == 'nn':
        model = NeuralNetwork(input_dim)

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
    print(classification_report(y_test, predictions))
    return cm

def train_model(data_dict, learning_rate=0.01, batch_size=128, num_epochs=50, 
                scheduler_step_size=5, scheduler_gamma=0.5, missing_value=-1,
                convergence_threshold=1e-4, patience=3, weight_decay=0,
                optimizer='sgd',
                model='lr'):  
    
    set_seeds()
    train_loader, test_loader = prepare_data_loaders(data_dict, batch_size, missing_value)
    input_dim = data_dict['X_train'].shape[1]
    
    
    model, criterion, optimizer, scheduler = initialize_model(
        input_dim, 
        learning_rate, 
        scheduler_step_size, 
        scheduler_gamma, 
        weight_decay=weight_decay,
        optimizer_name=optimizer,
        model=model
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