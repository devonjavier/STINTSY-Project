import numpy as np
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterSampler
import pandas as pd
import matplotlib.pyplot as plt


#optimized k value
def trainkNN(folds_data, k=9):
    accuracies_train = []
    losses_train = []
    knn_accuracies_test = []
    losses_test = []
    confusion_matrices = []
    
    all_y_test_aggregate = []
    all_predictions_aggregate = []

    for i, fold in enumerate(folds_data):
        print(f"fold {i+1}")

        X_train = fold['X_train']
        y_train = fold['y_train']
        X_test = fold['X_test']
        y_test = fold['y_test']

        knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_test)

        y_train_pred = knn_model.predict(X_train)
        y_train_pred_proba = knn_model.predict_proba(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)
        loss_train = log_loss(y_train, y_train_pred_proba)
 
        y_test_pred = knn_model.predict(X_test)
        y_test_pred_proba = knn_model.predict_proba(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        loss_test = log_loss(y_test, y_test_pred_proba)

        cm = confusion_matrix(y_test, y_test_pred)

        # Aggregate results
        all_y_test_aggregate.extend(y_test)
        all_predictions_aggregate.extend(y_test_pred)

        accuracies_train.append(accuracy_train)
        losses_train.append(loss_train)
        knn_accuracies_test.append(accuracy_test)
        losses_test.append(loss_test)
        confusion_matrices.append(cm)

        ## perf metrics
        print(f"Fold {i+1} - Train Accuracy: {accuracy_train:.4f}, Train Loss: {loss_train:.4f}, Test Accuracy: {accuracy_test:.4f}, Test Loss: {loss_test:.4f}")
        print(f"Fold {i+1} - Confusion Matrix:\n{cm}\n")

    # Calculate aggregate metrics
    print(f"{'='*20} AGGREGATE RESULTS {'='*20}")
    aggregate_cm = confusion_matrix(all_y_test_aggregate, all_predictions_aggregate)
    print("\nAggregate Confusion Matrix:")
    print(aggregate_cm)
    print("\nAggregate Classification Report:")
    print(classification_report(all_y_test_aggregate, all_predictions_aggregate, digits=8))

    # Calculate summary statistics for all folds
    avg_train_accuracy = np.mean(accuracies_train)
    avg_train_loss = np.mean(losses_train)
    avg_test_accuracy = np.mean(knn_accuracies_test)
    avg_test_loss = np.mean(losses_test)
    
    print("\nSummary Metrics:")
    print(f"Average Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Average Train Loss: {avg_train_loss:.4f}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Average Test Loss: {avg_test_loss:.4f}")

    return {
        'accuracies_train': accuracies_train,
        'losses_train': losses_train,
        'knn_accuracies_test': knn_accuracies_test,
        'losses_test': losses_test,
        'confusion_matrices': confusion_matrices,
        'aggregate_confusion_matrix': aggregate_cm,
        'aggregated_final_metrics': {
            'avg_final_train_loss': avg_train_loss,
            'avg_final_test_loss': avg_test_loss,
            'avg_final_train_accuracy': avg_train_accuracy,
            'avg_final_test_accuracy': avg_test_accuracy
        },
        'all_final_test_accuracies': knn_accuracies_test,
        'all_y_test_aggregate': all_y_test_aggregate,
        'all_predictions_aggregate': all_predictions_aggregate
    }

def hyperparameter_random_search_knn(folds_data, p_grid, n_iter_search=20, random_state=45):
    results = []
    param_list = list(ParameterSampler(p_grid, n_iter=n_iter_search, random_state=random_state))

    for params in param_list:
        print("\nTesting configuration:")
        print(params)
        
        fold_accuracies_train = []
        fold_losses_train = []
        fold_accuracies_test = []
        fold_losses_test = []
        fold_confusion_matrices = []
        
        all_y_test_aggregate = []
        all_predictions_aggregate = []

        for i in range(len(folds_data)):
            X_train, X_test, y_train, y_test = (
                folds_data[i]['X_train'],
                folds_data[i]['X_test'],
                folds_data[i]['y_train'],
                folds_data[i]['y_test'],
            )

            model = KNeighborsClassifier(**params)
            model.fit(X_train, y_train)

            # Train metrics
            y_train_pred = model.predict(X_train)
            y_train_pred_proba = model.predict_proba(X_train)
            accuracy_train = accuracy_score(y_train, y_train_pred)
            loss_train = log_loss(y_train, y_train_pred_proba)
            
            # Test metrics
            y_test_pred = model.predict(X_test)
            y_test_pred_proba = model.predict_proba(X_test)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            loss_test = log_loss(y_test, y_test_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            
            # Aggregate predictions for overall metrics
            all_y_test_aggregate.extend(y_test)
            all_predictions_aggregate.extend(y_test_pred)
            
            # Store fold results
            fold_accuracies_train.append(accuracy_train)
            fold_losses_train.append(loss_train)
            fold_accuracies_test.append(accuracy_test)
            fold_losses_test.append(loss_test)
            fold_confusion_matrices.append(cm)
            
            print(f"Fold {i+1} - Train Accuracy: {accuracy_train:.4f}, Train Loss: {loss_train:.4f}, Test Accuracy: {accuracy_test:.4f}, Test Loss: {loss_test:.4f}")
            print(f"Fold {i+1} - Confusion Matrix:\n{cm}\n")

        # Calculate averages
        avg_train_accuracy = np.mean(fold_accuracies_train)
        avg_train_loss = np.mean(fold_losses_train)
        avg_test_accuracy = np.mean(fold_accuracies_test)
        avg_test_loss = np.mean(fold_losses_test)
        
        # Calculate aggregate confusion matrix
        aggregate_cm = confusion_matrix(all_y_test_aggregate, all_predictions_aggregate)

        fold_results = []
        for i in range(len(folds_data)):
            fold_results.append({
                'train_accuracy': fold_accuracies_train[i],
                'train_loss': fold_losses_train[i],
                'test_accuracy': fold_accuracies_test[i],
                'test_loss': fold_losses_test[i],
                'confusion_matrix': fold_confusion_matrices[i]
            })

        results.append({
            'params': params,
            'test_accuracy': avg_test_accuracy,
            'test_loss': avg_test_loss,
            'final_train_loss': avg_train_loss,
            'final_test_loss': avg_test_loss,
            'final_train_accuracy': avg_train_accuracy,
            'final_test_accuracy': avg_test_accuracy,
            'fold_results': fold_results,
            'aggregate_confusion_matrix': aggregate_cm,
            'all_final_test_accuracies': fold_accuracies_test
        })

    # Sort results by test accuracy
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    best_params = results[0]['params']
    
    # Train final model with best parameters on all training data
    all_X_train = np.concatenate([folds_data[i]['X_train'] for i in range(len(folds_data))])
    all_y_train = np.concatenate([folds_data[i]['y_train'] for i in range(len(folds_data))])

    best_model = KNeighborsClassifier(**best_params)
    best_model.fit(all_X_train, all_y_train)

    print("\nBest Parameters:", best_params)
    print("Best Average Accuracy:", results[0]['test_accuracy'])
    
    # Create summary dataframe for visualization
    summary_df = pd.DataFrame([
        {
            'n_neighbors': r['params'].get('n_neighbors', 'N/A'),
            'weights': r['params'].get('weights', 'N/A'),
            'p': r['params'].get('p', 'N/A'),
            'Test Accuracy': r['test_accuracy'],
            'Test Loss': r['test_loss'],
            'Final Train Loss': r['final_train_loss'],
            'Final Test Loss': r['final_test_loss'],
            'Final Train Accuracy': r['final_train_accuracy'],
            'Final Test Accuracy': r['final_test_accuracy']
        } for r in results
    ])

    print("\nTop 10 Configurations:")
    print(summary_df.head(10))

    # Optional: Visualize results
    if 'n_neighbors' in p_grid:
        plt.figure(figsize=(12, 6))
        for weights in set(summary_df['weights']):
            subset = summary_df[summary_df['weights'] == weights]
            plt.scatter(subset['n_neighbors'], subset['Test Accuracy'], 
                       label=f"weights={weights}", alpha=0.7)
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Test Accuracy')
        plt.title('KNN Hyperparameter Search Results')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'best_model': best_model,
        'best_params': best_params,
        'results': results,
        'summary_df': summary_df,
        'all_final_test_accuracies': results[0]['all_final_test_accuracies']
    }

# Example usage:
# p_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11],
#     'weights': ['uniform', 'distance'],
#     'p': [1, 2], # Manhattan and Euclidean distances
# }
#
# results = hyperparameter_random_search_knn(folds_data, p_grid, n_iter_search=20)
# best_knn = results['best_model']
