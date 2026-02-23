"""
Generate synthetic training, test, and demo datasets for fraud detection.
This script creates PaySim-like transaction data and trains models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import preprocess_data, engineer_features, select_features
from models import HybridModel, save_models


def generate_transactions(n_samples=10000, fraud_ratio=0.1):
    """Generate synthetic transaction data"""
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    types = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    
    # Generate normal transactions
    normal_data = {
        'step': np.random.randint(0, 743, n_normal),
        'type': np.random.choice(types, n_normal),
        'amount': np.random.lognormal(8, 2, n_normal),
        'nameOrig': [f'C{i}' for i in np.random.randint(1000000, 9999999, n_normal)],
        'oldbalanceOrg': np.random.lognormal(10, 2, n_normal),
        'newbalanceOrig': np.random.lognormal(10, 2, n_normal),
        'nameDest': [f'C{i}' for i in np.random.randint(1000000, 9999999, n_normal)],
        'oldbalanceDest': np.random.lognormal(10, 2, n_normal),
        'newbalanceDest': np.random.lognormal(10, 2, n_normal),
        'isFraud': [0] * n_normal,
        'isFlaggedFraud': [0] * n_normal
    }
    
    # Generate fraudulent transactions (more extreme patterns)
    fraud_data = {
        'step': np.random.randint(0, 743, n_fraud),
        'type': np.random.choice(['TRANSFER', 'CASH-OUT'], n_fraud),
        'amount': np.random.lognormal(12, 1.5, n_fraud),  # Larger amounts
        'nameOrig': [f'C{i}' for i in np.random.randint(1000000, 9999999, n_fraud)],
        'oldbalanceOrg': np.random.lognormal(12, 1.5, n_fraud),
        'newbalanceOrig': np.zeros(n_fraud),  # Often drained
        'nameDest': [f'C{i}' for i in np.random.randint(1000000, 9999999, n_fraud)],
        'oldbalanceDest': np.random.lognormal(9, 2, n_fraud),
        'newbalanceDest': np.random.lognormal(12, 1.5, n_fraud),  # Large increases
        'isFraud': [1] * n_fraud,
        'isFlaggedFraud': [1] * n_fraud
    }
    
    # Combine and shuffle
    df_normal = pd.DataFrame(normal_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    print("=" * 60)
    print("Mobile Money Fraud Detection - Data Generation")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate datasets
    print("\n1. Generating training data (10,000 transactions)...")
    df_train = generate_transactions(n_samples=10000, fraud_ratio=0.1)
    df_train.to_csv('data/train_data.csv', index=False)
    print(f"   ✓ Saved to data/train_data.csv")
    
    print("\n2. Generating test data (500 transactions)...")
    df_test = generate_transactions(n_samples=500, fraud_ratio=0.15)
    df_test.to_csv('data/test_data.csv', index=False)
    print(f"   ✓ Saved to data/test_data.csv")
    
    print("\n3. Generating demo data (50 transactions)...")
    df_demo = generate_transactions(n_samples=50, fraud_ratio=0.2)
    df_demo.to_csv('data/demo_data.csv', index=False)
    print(f"   ✓ Saved to data/demo_data.csv")
    
    # Preprocess training data using NEW pipeline
    print("\n" + "=" * 60)
    print("Preprocessing Training Data with NEW Feature Engineering")
    print("=" * 60)
    
    # Drop fraud labels for unsupervised learning
    df_train_unsupervised = df_train.drop(columns=['isFraud', 'isFlaggedFraud'])
    
    # Apply NEW preprocessing pipeline
    X_train, scaler, df_processed = preprocess_data(df_train_unsupervised.copy())
    
    print(f"\n✓ Preprocessing complete:")
    print(f"   - Input shape: {X_train.shape}")
    print(f"   - Features generated: {df_processed.shape[1]}")
    print(f"   - Feature columns: {list(df_processed.columns)[:10]}...")
    
    # Verify One-Hot Encoding
    type_cols = [col for col in df_processed.columns if col.startswith('type_')]
    print(f"   - Type columns (One-Hot): {type_cols}")
    
    # Train models
    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)
    
    input_dim = X_train.shape[1]
    hybrid_model = HybridModel(input_dim=input_dim)
    
    print("\nTraining hybrid model ensemble...")
    hybrid_model.train(X_train, ae_epochs=30)
    
    # Save models
    print("\n" + "=" * 60)
    print("Saving Models")
    print("=" * 60)
    
    os.makedirs('trained_models', exist_ok=True)
    save_models(hybrid_model, scaler, filepath='trained_models')
    
    print("\n✓ Models saved to trained_models/:")
    print("   - scaler.pkl")
    print("   - isolation_forest.pkl")
    print("   - autoencoder.pth")
    print("   - dbscan.pkl")
    
    # Validate on test set
    print("\n" + "=" * 60)
    print("Validation on Test Set")
    print("=" * 60)
    
    df_test_unsupervised = df_test.drop(columns=['isFraud', 'isFlaggedFraud'])
    X_test, _, _ = preprocess_data(df_test_unsupervised.copy(), scaler=scaler)
    
    predictions, hybrid_scores, individual_scores = hybrid_model.predict(X_test)
    
    print(f"\n✓ Test results:")
    print(f"   - Total transactions: {len(predictions)}")
    print(f"   - Flagged as anomalies: {predictions.sum()}")
    print(f"   - Hybrid score range: [{hybrid_scores.min():.3f}, {hybrid_scores.max():.3f}]")
    print(f"   - Default threshold: {hybrid_model.default_threshold:.3f}")
    
    # Show actual fraud detection performance (if labels available)
    if 'isFraud' in df_test.columns:
        true_frauds = df_test['isFraud'].values
        detected_frauds = predictions == 1
        
        tp = ((detected_frauds == 1) & (true_frauds == 1)).sum()
        fp = ((detected_frauds == 1) & (true_frauds == 0)).sum()
        fn = ((detected_frauds == 0) & (true_frauds == 1)).sum()
        tn = ((detected_frauds == 0) & (true_frauds == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n   Performance Metrics:")
        print(f"   - Precision: {precision:.2%}")
        print(f"   - Recall: {recall:.2%}")
        print(f"   - True Positives: {tp}")
        print(f"   - False Positives: {fp}")
        print(f"   - False Negatives: {fn}")
        print(f"   - True Negatives: {tn}")
    
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: streamlit run app.py")
    print("2. Login with username: admin, password: admin123")
    print("3. Upload data/test_data.csv in Real-Time Monitor or Batch Analysis")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
