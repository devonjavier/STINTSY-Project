from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import ParameterSampler
def train_eval_dt(folds_data):
    
    accuracies_train = []
    losses_train = []
    dt_accuracies_test = []
    losses_test = []
    confusion_matrices = []
    all_test_predictions = [] # Store predictions for all folds
    all_test_labels = [] # Store true labels for all folds

    for i in range(5):
        X_train, X_test, y_train, y_test = (
            folds_data[i]['X_train'],
            folds_data[i]['X_test'],
            folds_data[i]['y_train'],
            folds_data[i]['y_test'],
        )

        model = DecisionTreeClassifier(random_state=45, min_impurity_decrease=0.001)

        model.fit(X_train, y_train)

        # Train predictions and loss
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)
        loss_train = log_loss(y_train, y_train_pred_proba)

        # Test predictions and loss
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        loss_test = log_loss(y_test, y_test_pred_proba)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        confusion_matrices.append(cm)

        accuracies_train.append(accuracy_train)
        losses_train.append(loss_train)
        dt_accuracies_test.append(accuracy_test)
        losses_test.append(loss_test)
        all_test_predictions.append(y_test_pred)
        all_test_labels.append(y_test)

        print(f"Fold {i+1} - Train Accuracy: {accuracy_train:.4f}, Train Loss: {loss_train:.4f}, Test Accuracy: {accuracy_test:.4f}, Test Loss: {loss_test:.4f}")
        print(f"Fold {i+1} - Confusion Matrix:\n{cm}\n")

    avg_accuracy_train = np.mean(accuracies_train)
    avg_loss_train = np.mean(losses_train)
    avg_accuracy_test = np.mean(dt_accuracies_test)
    avg_loss_test = np.mean(losses_test)

    print(f"\nAggregated Metrics:")
    print(f"Average Train Accuracy: {avg_accuracy_train:.4f}")
    print(f"Average Train Loss: {avg_loss_train:.4f}")
    print(f"Average Test Accuracy: {avg_accuracy_test:.4f}")
    print(f"Average Test Loss: {avg_loss_test:.4f}")

    print("\nOverall Confusion Matrix:")
    print(sum(confusion_matrices))

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    return avg_accuracy_train, avg_loss_train, avg_accuracy_test, avg_loss_test, all_test_predictions, all_test_labels, X_test, y_test, X_train, y_train

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterSampler

def hyperparameter_random_search_dt(folds_data, p_grid, n_iter_search=50, random_state=45):
    results = []

    # Generate random parameter combinations
    param_list = list(ParameterSampler(p_grid, n_iter=n_iter_search, random_state=random_state))

    for params in param_list:
        fold_accuracies = []
        fold_losses = []
        fold_test_accuracies = [] # List to store test accuracies per fold

        # Evaluate parameters across all folds
        for i in range(len(folds_data)):
            X_train, X_test, y_train, y_test = (
                folds_data[i]['X_train'],
                folds_data[i]['X_test'],
                folds_data[i]['y_train'],
                folds_data[i]['y_test'],
            )

            model = DecisionTreeClassifier(random_state=random_state, **params)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            y_test_pred_proba = model.predict_proba(X_test)

            fold_accuracies.append(accuracy_score(y_test, y_test_pred))
            fold_losses.append(log_loss(y_test, y_test_pred_proba))
            fold_test_accuracies.append(accuracy_score(y_test, y_test_pred)) # Save the test accuracy for this fold.

        # Calculate average performance across folds
        avg_accuracy = np.mean(fold_accuracies)
        avg_loss = np.mean(fold_losses)

        results.append({
            'params': params,
            'avg_accuracy': avg_accuracy,
            'avg_loss': avg_loss,
            'fold_test_accuracies': fold_test_accuracies # Add fold test accuracies to results
        })

    # Select best parameters based on average accuracy
    results.sort(key=lambda x: x['avg_accuracy'], reverse=True)
    best_params = results[0]['params']

    # Train final model with best parameters on all training data
    all_X_train = np.concatenate([folds_data[i]['X_train'] for i in range(len(folds_data))]) #corrected range.
    all_y_train = np.concatenate([folds_data[i]['y_train'] for i in range(len(folds_data))]) #corrected range.

    best_model = DecisionTreeClassifier(random_state=random_state, **best_params)
    best_model.fit(all_X_train, all_y_train)

    print("\nBest Parameters:", best_params)
    print("Best Average Accuracy:", results[0]['avg_accuracy'])

    return {
        'best_model': best_model,
        'best_params': best_params,
        'results': results,
        'fold_test_accuracies': results[0]['fold_test_accuracies'] #add fold accuracies to return.
    }