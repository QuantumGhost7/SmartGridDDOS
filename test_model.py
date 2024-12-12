import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, scaler, test_data):
    """Evaluate model performance on test data"""
    
    # Prepare features
    X_test = test_data[['packets_per_second', 'speed_mbps', 'packet_size']]
    y_test = test_data['is_ddos']
    
    # Scale features using the same scaler used during training
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_results(results, test_data):
    """Create visualizations for model performance"""
    
    plt.figure(figsize=(15, 10))
    
    # 1. Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(results['y_test'], results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Data)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 2. ROC Curve
    plt.subplot(2, 2, 2)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'][:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Data)')
    plt.legend(loc="lower right")
    
    # 3. Feature Distribution
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=test_data, x='packets_per_second', y='speed_mbps', 
                    hue='is_ddos', alpha=0.5)
    plt.title('Feature Distribution (Test Data)')
    
    # 4. Prediction Probability Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(results['y_pred_proba'][:, 1], bins=50)
    plt.title('Prediction Probability Distribution (Test Data)')
    plt.xlabel('Probability of DDoS')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('test_evaluation_results.png')
    plt.close()

def main():
    # Load model and scaler
    print("Loading model and scaler...")
    model = joblib.load('ddos_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv('test_data.csv')
    
    # Evaluate model
    print("Evaluating model on test data...")
    results = evaluate_model(model, scaler, test_data)
    
    # Print results
    print("\nModel Performance Metrics on Test Data:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    # Create visualizations
    print("\nGenerating visualization plots...")
    plot_results(results, test_data)
    print("Plots saved as 'test_evaluation_results.png'")
    
    # Save test results
    test_data['predicted'] = results['y_pred']
    test_data['prediction_probability'] = results['y_pred_proba'][:, 1]
    test_data.to_csv('test_results.csv', index=False)
    print("\nTest results saved to 'test_results.csv'")

if __name__ == "__main__":
    main()