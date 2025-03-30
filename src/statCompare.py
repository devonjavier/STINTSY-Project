import numpy as np
import pandas as pd
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman

def compare_models_with_friedman(model_results):

    # Extract model names and accuracies
    model_names = list(model_results.keys())
    accuracies = list(model_results.values())
    
    # Create a DataFrame for the comparison
    # Each row is a fold, each column is a model
    df = pd.DataFrame(accuracies).T
    df.columns = model_names
    
    # Perform Friedman test
    friedman_result = stats.friedmanchisquare(*[df[model] for model in model_names])
    
    print(f"Friedman test statistic: {friedman_result.statistic:.4f}")
    print(f"p-value: {friedman_result.pvalue:.4f}")
    
    if friedman_result.pvalue < 0.05:
        print("There are significant differences between models.")
        
        # Perform post-hoc Nemenyi test
        posthoc_result = posthoc_nemenyi_friedman(df)
        print("\nPost-hoc Nemenyi test p-values:")
        print(posthoc_result)
        
        # Rank the models
        ranks = df.rank(axis=1, ascending=False)
        mean_ranks = ranks.mean()
        print("\nMean ranks (lower is better):")
        for model, rank in mean_ranks.items():
            print(f"{model}: {rank:.4f}")
        
        # Find the best model based on mean rank
        best_model = mean_ranks.idxmin()
        print(f"\nBest model based on ranks: {best_model}")
    else:
        print("No significant differences detected between models.")
    
    return {
        'friedman_result': friedman_result,
        'df': df,
        'mean_ranks': mean_ranks if friedman_result.pvalue < 0.05 else None,
        'posthoc_result': posthoc_result if friedman_result.pvalue < 0.05 else None
    }

