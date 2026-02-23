"""
Models module for Mobile Money Fraud Detection
Implements Isolation Forest, Autoencoder, and DBSCAN with hybrid scoring
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from typing import Tuple, Dict


class Autoencoder(nn.Module):
    """
    PyTorch Autoencoder for anomaly detection.
    High reconstruction error indicates anomalies.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 10):
        """
        Initialize the Autoencoder.

        Args:
            input_dim: Number of input features
            encoding_dim: Size of the encoded representation
        """
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderModel:
    """Wrapper class for Autoencoder training and inference."""

    def __init__(self, input_dim: int, encoding_dim: int = 10):
        self.model = Autoencoder(input_dim, encoding_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

    def train(
        self, X_train: np.ndarray, epochs: int = 50, learning_rate: float = 0.001
    ):
        """
        Train the autoencoder.

        Args:
            X_train: Training data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        X_tensor = torch.FloatTensor(X_train)

        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, X_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Autoencoder Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}"
                )

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction errors for anomaly detection.

        Args:
            X: Input data

        Returns:
            Array of reconstruction errors
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            return errors.numpy()

    def predict(self, X: np.ndarray, threshold_percentile: float = 95) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error.

        Args:
            X: Input data
            threshold_percentile: Percentile for anomaly threshold

        Returns:
            Binary predictions (1 for anomaly, 0 for normal)
        """
        errors = self.get_reconstruction_errors(X)
        threshold = np.percentile(errors, threshold_percentile)
        return (errors > threshold).astype(int)


class IsolationForestModel:
    """Wrapper for Sklearn's Isolation Forest."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest.

        Args:
            contamination: Expected proportion of outliers
            random_state: Random seed
        """
        self.model = IsolationForest(
            contamination=contamination, random_state=random_state, n_estimators=100
        )

    def train(self, X_train: np.ndarray):
        """
        Train the Isolation Forest.

        Args:
            X_train: Training data
        """
        self.model.fit(X_train)
        print("Isolation Forest training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (-1 for anomaly, 1 for normal).
        Converts to (1 for anomaly, 0 for normal).

        Args:
            X: Input data

        Returns:
            Binary predictions
        """
        predictions = self.model.predict(X)
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        return (predictions == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower score = more anomalous).

        Args:
            X: Input data

        Returns:
            Anomaly scores
        """
        return self.model.score_samples(X)


class DBSCANModel:
    """Wrapper for Sklearn's DBSCAN clustering."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize DBSCAN.

        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = None
        self.training_data = None
        self.eps = eps

    def train(self, X_train: np.ndarray):
        """
        Fit DBSCAN (no training, just clustering).

        Args:
            X_train: Training data
        """
        self.training_data = X_train
        self.labels_ = self.model.fit_predict(X_train)
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        print(f"DBSCAN complete: {n_clusters} clusters, {n_noise} noise points")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using distance to nearest training sample.
        If no training data exists, fall back to fit_predict.

        Args:
            X: Input data

        Returns:
            Binary predictions (1 for anomaly/noise, 0 for normal)
        """
        if self.training_data is None:
            # Fallback: re-fit if no training data (legacy behavior)
            labels = self.model.fit_predict(X)
            return (labels == -1).astype(int)

        # Calculate minimum distance to training samples for each point
        from scipy.spatial.distance import cdist

        distances = cdist(X, self.training_data, metric="euclidean")
        min_distances = np.min(distances, axis=1)

        # Points far from training clusters (> eps) are anomalies
        predictions = (min_distances > self.eps).astype(int)
        return predictions




class HybridModel:
    """
    Hybrid ensemble combining Isolation Forest, Autoencoder, and DBSCAN.
    """

    def __init__(self, input_dim: int):
        """
        Initialize all three models.

        Args:
            input_dim: Number of input features
        """
        self.iso_forest = IsolationForestModel(contamination=0.1)
        self.autoencoder = AutoencoderModel(input_dim=input_dim)
        self.dbscan = DBSCANModel(eps=0.5, min_samples=5)
        self.input_dim = input_dim
        self.iso_score_bounds = None
        self.ae_error_bounds = None
        self.default_threshold = 0.21  # Adjusted for normalized 0-1 scale

    def train(self, X_train: np.ndarray, ae_epochs: int = 50):
        """
        Train all three models in parallel.

        Args:
            X_train: Training data
            ae_epochs: Epochs for autoencoder training
        """
        print("Training Isolation Forest...")
        self.iso_forest.train(X_train)

        print("\nTraining Autoencoder...")
        self.autoencoder.train(X_train, epochs=ae_epochs)

        print("\nFitting DBSCAN...")
        self.dbscan.train(X_train)

        # Calculate raw scores on training data for threshold calibration
        iso_train_scores = self.iso_forest.score_samples(X_train)
        ae_train_errors = self.autoencoder.get_reconstruction_errors(X_train)
        
        # Store bounds for reference only
        self.iso_score_bounds = (
            float(np.min(iso_train_scores)),
            float(np.max(iso_train_scores)),
        )
        self.ae_error_bounds = (
            float(np.min(ae_train_errors)),
            float(np.max(ae_train_errors)),
        )

        # Calculate normalized hybrid scores on training data
        hybrid_train_scores, _ = self.calculate_hybrid_score(X_train)
        
        # Set default threshold at 75th percentile of normalized training scores
        self.default_threshold = float(np.percentile(hybrid_train_scores, 75))

        print(f"\nAll models trained successfully!")
        print(f"Default threshold set to: {self.default_threshold:.4f}")
        print(f"Training AE error range: [{self.ae_error_bounds[0]:.2e}, {self.ae_error_bounds[1]:.2e}]")
        print(f"Training ISO score range: [{self.iso_score_bounds[0]:.4f}, {self.iso_score_bounds[1]:.4f}]")

    def calculate_hybrid_score(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Calculate hybrid anomaly scores using LOG TRANSFORMATION and FIXED NORMALIZATION.

        Steps:
        1. Tame Autoencoder MSE with log1p transformation
        2. Normalize to 0-1 using fixed constants (NOT batch-dependent)
        3. Combine with weighted average
        
        Args:
            X: Input data

        Returns:
            Tuple of (hybrid_scores, individual_scores_dict)
        """
        # Get raw individual scores
        iso_raw = self.iso_forest.score_samples(X)  # Negative scores, more negative = more anomalous
        ae_raw_mse = self.autoencoder.get_reconstruction_errors(X)  # Raw MSE, can explode
        dbscan_preds = self.dbscan.predict(X)  # Binary: 1 = anomaly, 0 = normal

        # ===== STEP 1: TAME THE AUTOENCODER =====
        # Apply log transformation to compress explosive MSE values
        ae_log = np.log1p(ae_raw_mse)  # log(1 + x) to handle zeros safely
        
        # ===== STEP 2: FIXED NORMALIZATION (NOT BATCH-DEPENDENT) =====
        # Normalize logged AE scores to ~0-1 range using fixed constant
        # Empirically calibrated: divide by 25 works for typical log(MSE) ranges
        ae_normalized = np.clip(ae_log / 25.0, 0, 1)
        
        # Normalize Isolation Forest scores to 0-1 range
        # ISO scores are typically around -0.5 to 0.5, negate and shift
        iso_normalized = np.clip(-iso_raw, 0, 1)  # Negate so higher = more anomalous
        
        # DBSCAN is already binary 0/1, use as-is
        dbscan_normalized = dbscan_preds.astype(float)

        # ===== STEP 3: COMBINE WITH WEIGHTED AVERAGE =====
        # Use 50-50 split between ISO and AE, with small DBSCAN contribution
        hybrid_scores = (
            0.45 * iso_normalized + 
            0.45 * ae_normalized + 
            0.10 * dbscan_normalized
        )

        # Store individual normalized scores for debugging
        individual_scores = {
            "isolation_forest": iso_normalized,
            "autoencoder": ae_normalized,
            "dbscan": dbscan_normalized,
        }

        return hybrid_scores, individual_scores

    def predict(
        self, X: np.ndarray, threshold: float = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict anomalies using normalized hybrid scoring.

        Args:
            X: Input data
            threshold: Threshold for blocking (uses calibrated default if None)

        Returns:
            Tuple of (predictions, hybrid_scores, individual_scores)
        """
        if threshold is None:
            threshold = self.default_threshold
            
        hybrid_scores, individual_scores = self.calculate_hybrid_score(X)
        predictions = (hybrid_scores > threshold).astype(int)

        return predictions, hybrid_scores, individual_scores




def save_models(hybrid_model: HybridModel, scaler, filepath: str = "trained_models"):
    """
    Save trained models and scaler to disk.

    Args:
        hybrid_model: Trained HybridModel instance
        scaler: Fitted scaler object
        filepath: Directory path to save models
    """
    os.makedirs(filepath, exist_ok=True)

    # Save Isolation Forest
    joblib.dump(hybrid_model.iso_forest, os.path.join(filepath, "isolation_forest.pkl"))

    # Save Autoencoder
    torch.save(
        {
            "model_state_dict": hybrid_model.autoencoder.model.state_dict(),
            "input_dim": hybrid_model.autoencoder.input_dim,
            "encoding_dim": hybrid_model.autoencoder.encoding_dim,
        },
        os.path.join(filepath, "autoencoder.pth"),
    )

    # Save DBSCAN
    joblib.dump(hybrid_model.dbscan, os.path.join(filepath, "dbscan.pkl"))

    # Save scaler
    joblib.dump(scaler, os.path.join(filepath, "scaler.pkl"))

    # Save normalization calibration metadata
    calibration = {
        "iso_score_bounds": hybrid_model.iso_score_bounds,
        "ae_error_bounds": hybrid_model.ae_error_bounds,
        "default_threshold": hybrid_model.default_threshold,
    }
    joblib.dump(calibration, os.path.join(filepath, "calibration.pkl"))

    print(f"Models saved to {filepath}/")


def load_models(filepath: str = "trained_models") -> Tuple[HybridModel, object]:
    """
    Load trained models and scaler from disk.

    Args:
        filepath: Directory path containing saved models

    Returns:
        Tuple of (HybridModel, scaler)
    """
    # Load scaler first to get input dimension
    scaler = joblib.load(os.path.join(filepath, "scaler.pkl"))

    # Load Autoencoder checkpoint
    ae_checkpoint = torch.load(os.path.join(filepath, "autoencoder.pth"))
    input_dim = ae_checkpoint["input_dim"]

    # Create HybridModel instance
    hybrid_model = HybridModel(input_dim=input_dim)

    # Load Isolation Forest
    hybrid_model.iso_forest = joblib.load(
        os.path.join(filepath, "isolation_forest.pkl")
    )

    # Load Autoencoder state
    hybrid_model.autoencoder.model.load_state_dict(ae_checkpoint["model_state_dict"])
    hybrid_model.autoencoder.model.eval()

    # Load DBSCAN
    hybrid_model.dbscan = joblib.load(os.path.join(filepath, "dbscan.pkl"))

    # Load optional calibration metadata (for backward compatibility)
    calibration_path = os.path.join(filepath, "calibration.pkl")
    if os.path.exists(calibration_path):
        calibration = joblib.load(calibration_path)
        hybrid_model.iso_score_bounds = calibration.get("iso_score_bounds")
        hybrid_model.ae_error_bounds = calibration.get("ae_error_bounds")
        hybrid_model.default_threshold = float(
            calibration.get("default_threshold", hybrid_model.default_threshold)
        )

    print(f"Models loaded from {filepath}/")

    return hybrid_model, scaler
