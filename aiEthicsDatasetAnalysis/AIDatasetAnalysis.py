import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load COMPAS dataset (you'll need to download this)
# Dataset available at: https://github.com/propublica/compas-analysis
def load_compas_data():
    # This is a placeholder - you'll need to load the actual COMPAS data
    # The actual dataset should include columns like:
    # 'race', 'sex', 'age', 'priors_count', 'decile_score', 'two_year_recid'
    
    # Sample structure (replace with actual data loading):
    # df = pd.read_csv('compas-scores-two-years.csv')
    
    # For demonstration, creating a sample structure
    print("Please load the actual COMPAS dataset from:")
    print("https://github.com/propublica/compas-analysis")
    return None

def analyze_bias(df):
    """Analyze racial bias in COMPAS dataset"""
    
    # Convert to AIF360 StandardDataset
    dataset = StandardDataset(
        df=df,
        label_name='two_year_recid',
        favorable_classes=[0],  # 0 = no recidivism
        protected_attribute_names=['race'],
        privileged_classes=[['Caucasian']]  # Caucasian as privileged group
    )
    
    # Split data
    dataset_train, dataset_test = dataset.split([0.7], shuffle=True)
    
    # Calculate bias metrics
    metric = BinaryLabelDatasetMetric(
        dataset_test, 
        unprivileged_groups=[{'race': 1}],  # African-American as unprivileged
        privileged_groups=[{'race': 0}]
    )
    
    print("Bias Metrics:")
    print(f"Disparate Impact: {metric.disparate_impact():.3f}")
    print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.3f}")
    
    # Train a simple classifier
    model = LogisticRegression()
    model.fit(dataset_train.features, dataset_train.labels.ravel())
    
    # Predictions
    pred = model.predict(dataset_test.features)
    
    # Calculate classification metrics by race
    return calculate_race_metrics(dataset_test, pred)

def calculate_race_metrics(dataset, predictions):
    """Calculate metrics by racial group"""
    
    # Separate by race (simplified - African-American vs Caucasian)
    aa_mask = dataset.protected_attributes.ravel() == 1  # African-American
    ca_mask = dataset.protected_attributes.ravel() == 0  # Caucasian
    
    aa_labels = dataset.labels[aa_mask]
    ca_labels = dataset.labels[ca_mask]
    aa_pred = predictions[aa_mask]
    ca_pred = predictions[ca_mask]
    
    # Calculate metrics
    metrics = {}
    
    # False Positive Rates
    aa_fp = np.sum((aa_pred == 1) & (aa_labels == 0)) / np.sum(aa_labels == 0)
    ca_fp = np.sum((ca_pred == 1) & (ca_labels == 0)) / np.sum(ca_labels == 0)
    
    metrics['fpr_aa'] = aa_fp
    metrics['fpr_ca'] = ca_fp
    metrics['fpr_ratio'] = aa_fp / ca_fp
    
    # False Negative Rates
    aa_fn = np.sum((aa_pred == 0) & (aa_labels == 1)) / np.sum(aa_labels == 1)
    ca_fn = np.sum((ca_pred == 0) & (ca_labels == 1)) / np.sum(ca_labels == 1)
    
    metrics['fnr_aa'] = aa_fn
    metrics['fnr_ca'] = ca_fn
    metrics['fnr_ratio'] = aa_fn / ca_fn
    
    return metrics

def create_visualizations(metrics):
    """Create bias visualization charts"""
    
    # False Positive Rate Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # FPR Comparison
    races = ['African-American', 'Caucasian']
    fpr_values = [metrics['fpr_aa'], metrics['fpr_ca']]
    
    ax1.bar(races, fpr_values, color=['#ff6b6b', '#4ecdc4'])
    ax1.set_title('False Positive Rates by Race')
    ax1.set_ylabel('False Positive Rate')
    ax1.set_ylim(0, max(fpr_values) * 1.2)
    
    for i, v in enumerate(fpr_values):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # FPR Ratio
    ax2.bar(['FPR Ratio\n(AA vs Caucasian)'], [metrics['fpr_ratio']], 
           color='orange' if metrics['fpr_ratio'] > 1 else 'green')
    ax2.set_title('False Positive Rate Ratio')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Ratio')
    
    plt.tight_layout()
    plt.savefig('compas_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def apply_bias_mitigation(df):
    """Apply reweighing bias mitigation technique"""
    
    dataset = StandardDataset(
        df=df,
        label_name='two_year_recid',
        favorable_classes=[0],
        protected_attribute_names=['race'],
        privileged_classes=[['Caucasian']]
    )
    
    # Apply reweighing
    RW = Reweighing(unprivileged_groups=[{'race': 1}],
                   privileged_groups=[{'race': 0}])
    dataset_transformed = RW.fit_transform(dataset)
    
    return dataset_transformed

# Main execution
if __name__ == "__main__":
    # Load your COMPAS dataset here
    # df = load_compas_data()
    
    # For demonstration, we'll create sample output
    print("COMPAS Dataset Bias Analysis")
    print("=" * 50)
    
    # Sample metrics (replace with actual analysis)
    sample_metrics = {
        'fpr_aa': 0.45,  # African-American FPR
        'fpr_ca': 0.23,  # Caucasian FPR
        'fpr_ratio': 1.96,  # FPR ratio
        'fnr_aa': 0.28,   # African-American FNR
        'fnr_ca': 0.31,   # Caucasian FNR
        'fnr_ratio': 0.90  # FNR ratio
    }
    
    print("Key Findings:")
    print(f"False Positive Rate - African-American: {sample_metrics['fpr_aa']:.3f}")
    print(f"False Positive Rate - Caucasian: {sample_metrics['fpr_ca']:.3f}")
    print(f"FPR Ratio (AA/Caucasian): {sample_metrics['fpr_ratio']:.2f}")
    print(f"Disparate Impact: {0.67:.2f}")  # Sample value
    
    create_visualizations(sample_metrics)