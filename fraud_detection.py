#!/usr/bin/env python3
"""
Simple Fraud Detection using PyOD AutoEncoder
A clean, easy-to-use implementation for credit card fraud detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pyod.models.auto_encoder import AutoEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load the credit card dataset and show basic information."""
    print("📊 Loading Credit Card Dataset...")
    
    # Load the data
    data = pd.read_csv('creditcard.csv')
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.shape[1] - 1}")  # Excluding 'Class' column
    print(f"Transactions: {data.shape[0]:,}")
    
    # Show fraud distribution
    fraud_count = data['Class'].sum()
    total_count = len(data)
    fraud_percentage = (fraud_count / total_count) * 100
    
    print(f"\nFraud Detection Challenge:")
    print(f"✅ Legitimate transactions: {total_count - fraud_count:,}")
    print(f"🚨 Fraudulent transactions: {fraud_count:,}")
    print(f"📈 Fraud rate: {fraud_percentage:.3f}%")
    
    return data

def prepare_data(data):
    """Prepare data for the AutoEncoder model."""
    print("\n🔧 Preparing Data...")
    
    # Separate features (X) and target (y)
    X = data.drop('Class', axis=1)  # All columns except 'Class'
    y = data['Class']               # Only the 'Class' column
    
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    # Scale the features (important for neural networks!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Data scaled successfully")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def create_and_train_autoencoder(X_train_scaled):
    """Create and train the AutoEncoder model."""
    print("\n🧠 Creating AutoEncoder Model...")
    
    # Create the AutoEncoder with basic PyOD parameters
    autoencoder = AutoEncoder(
        random_state=42,                      # For reproducible results
        verbose=1                             # Show training progress
    )
    
    print("AutoEncoder Architecture:")
    print(f"  Input: {X_train_scaled.shape[1]} features")
    print(f"  Using default PyOD AutoEncoder configuration")
    print(f"  Output: {X_train_scaled.shape[1]} features")
    
    print("\n🏋️ Training AutoEncoder...")
    print("This will take a few minutes...")
    
    # Train the model
    autoencoder.fit(X_train_scaled)
    
    print("✅ Training completed!")
    
    return autoencoder

def detect_fraud(autoencoder, X_test_scaled, y_test):
    """Detect fraud using the trained AutoEncoder."""
    print("\n🔍 Detecting Fraud...")
    
    # Get reconstruction errors (anomaly scores)
    test_scores = autoencoder.decision_function(X_test_scaled)
    
    print(f"Reconstruction error range: {test_scores.min():.4f} to {test_scores.max():.4f}")
    
    # Find optimal threshold (95th percentile of training scores)
    train_scores = autoencoder.decision_scores_
    threshold = np.percentile(train_scores, 95)
    
    print(f"Fraud threshold: {threshold:.4f}")
    
    # Make predictions
    predictions = (test_scores > threshold).astype(int)
    
    # Calculate performance metrics
    accuracy = (predictions == y_test).mean()
    fraud_detected = predictions.sum()
    actual_fraud = y_test.sum()
    
    print(f"\n📊 Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Fraud detected: {fraud_detected}")
    print(f"  Actual fraud: {actual_fraud}")
    
    return predictions, test_scores, threshold

def evaluate_model(y_test, predictions, test_scores):
    """Evaluate the model performance."""
    print("\n📈 Model Evaluation...")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Legitimate', 'Fraud']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Legit  Fraud")
    print(f"Actual Legit    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"      Fraud     {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, test_scores)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    return cm, roc_auc

def visualize_results(y_test, test_scores, threshold):
    """Create visualizations of the results."""
    print("\n📊 Creating Visualizations...")
    
    # Create a figure with multiple plots
    plt.figure(figsize=(15, 10))
    
    # 1. Score distribution
    plt.subplot(2, 3, 1)
    plt.hist(test_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Reconstruction Error (Anomaly Score)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    
    # 2. Score distribution by class
    plt.subplot(2, 3, 2)
    legitimate_scores = test_scores[y_test == 0]
    fraud_scores = test_scores[y_test == 1]
    
    plt.hist(legitimate_scores, bins=30, alpha=0.7, label='Legitimate', color='green')
    plt.hist(fraud_scores, bins=30, alpha=0.7, label='Fraud', color='red')
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Scores by Transaction Type')
    plt.legend()
    
    # 3. Fraud detection results - Fixed to handle edge cases
    plt.subplot(2, 3, 3)
    predictions = (test_scores > threshold).astype(int)
    
    # Calculate components safely
    fraud_detected = (predictions * y_test).sum()  # True positives
    fraud_missed = y_test.sum() - fraud_detected   # False negatives
    legitimate_correct = len(y_test) - y_test.sum() - (predictions * (1 - y_test)).sum()  # True negatives
    false_positives = predictions.sum() - fraud_detected  # False positives
    
    # Ensure all values are non-negative
    fraud_detected = max(0, fraud_detected)
    fraud_missed = max(0, fraud_missed)
    legitimate_correct = max(0, legitimate_correct)
    false_positives = max(0, false_positives)
    
    # Only create pie chart if we have valid data
    if fraud_detected + fraud_missed + legitimate_correct + false_positives > 0:
        labels = ['Fraud Detected', 'Fraud Missed', 'Legitimate (Correct)', 'False Positives']
        sizes = [fraud_detected, fraud_missed, legitimate_correct, false_positives]
        colors = ['red', 'orange', 'green', 'yellow']
        
        # Filter out zero values to avoid pie chart issues
        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
        if non_zero_indices:
            filtered_labels = [labels[i] for i in non_zero_indices]
            filtered_sizes = [sizes[i] for i in non_zero_indices]
            filtered_colors = [colors[i] for i in non_zero_indices]
            
            plt.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
                   autopct='%1.1f%%', startangle=90)
            plt.title('Fraud Detection Results')
        else:
            plt.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Fraud Detection Results')
    else:
        plt.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Fraud Detection Results')
    
    # 4. ROC Curve
    plt.subplot(2, 3, 4)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, test_scores)
    roc_auc = roc_auc_score(y_test, test_scores)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    # 5. Threshold analysis
    plt.subplot(2, 3, 5)
    thresholds = np.percentile(test_scores, np.arange(90, 100, 1))
    fraud_detected_at_threshold = []
    
    for thresh in thresholds:
        pred = (test_scores > thresh).astype(int)
        fraud_detected_at_threshold.append(pred.sum())
    
    plt.plot(thresholds, fraud_detected_at_threshold, 'b-', linewidth=2)
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Selected: {threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Fraud Detected')
    plt.title('Threshold vs Fraud Detection')
    plt.legend()
    plt.grid(True)
    
    # 6. Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calculate key metrics safely
    predictions = (test_scores > threshold).astype(int)
    accuracy = (predictions == y_test).mean()
    
    # Handle division by zero cases
    if predictions.sum() > 0:
        precision = (predictions * y_test).sum() / predictions.sum()
    else:
        precision = 0.0
        
    if y_test.sum() > 0:
        recall = (predictions * y_test).sum() / y_test.sum()
    else:
        recall = 0.0
    
    summary_text = f"""
    MODEL PERFORMANCE SUMMARY
    
    📊 Accuracy: {accuracy:.4f}
    🎯 Precision: {precision:.4f}
    📈 Recall: {recall:.4f}
    🚨 Fraud Detected: {predictions.sum()}
    ✅ Actual Fraud: {y_test.sum()}
    🎭 Threshold: {threshold:.4f}
    """
    
    plt.text(0.1, 0.5, summary_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizations created successfully!")

def save_model(autoencoder, scaler, threshold, filename='fraud_detection_model.pkl'):
    """Save the trained model for future use."""
    print(f"\n💾 Saving Model to {filename}...")
    
    import joblib
    
    model_data = {
        'autoencoder': autoencoder,
        'scaler': scaler,
        'threshold': threshold,
        'feature_count': scaler.n_features_in_
    }
    
    joblib.dump(model_data, filename)
    print(f"✅ Model saved successfully!")

def main():
    """Main function to run the complete fraud detection pipeline."""
    print("🚀 CREDIT CARD FRAUD DETECTION WITH AUTOENCODER")
    print("=" * 60)
    
    try:
        # Step 1: Load and explore data
        data = load_and_explore_data()
        
        # Step 2: Prepare data
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(data)
        
        # Step 3: Create and train AutoEncoder
        autoencoder = create_and_train_autoencoder(X_train_scaled)
        
        # Step 4: Detect fraud
        predictions, test_scores, threshold = detect_fraud(autoencoder, X_test_scaled, y_test)
        
        # Step 5: Evaluate model
        cm, roc_auc = evaluate_model(y_test, predictions, test_scores)
        
        # Step 6: Visualize results
        visualize_results(y_test, test_scores, threshold)
        
        # Step 7: Save model
        save_model(autoencoder, scaler, threshold)
        
        print("\n🎉 FRAUD DETECTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Your model is ready to detect fraud in new transactions!")
        
    except FileNotFoundError:
        print("❌ Error: 'creditcard.csv' file not found!")
        print("Please make sure the file is in the same directory as this script.")
        print("You can download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Please check your data and try again.")

if __name__ == "__main__":
    main() 