"""
Trained Evaluator Model Inference
===================================
Loads and uses trained ML models for prediction.

Provides predictions for:
- Relevance score
- Hallucination likelihood
- Faithfulness score
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TrainedEvaluator:
    """
    Loads and uses trained evaluator models for predictions.
    Gracefully handles missing models.
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize trained evaluator by loading models.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.available_models = []
        
        self._load_models()
        logger.info(f"Trained evaluator initialized with {len(self.available_models)} models")

    def _load_models(self) -> None:
        """Load all available trained models and scaler."""
        model_files = {
            'relevance': self.model_dir / 'relevance_model.pkl',
            'hallucination': self.model_dir / 'hallucination_model.pkl',
            'faithfulness': self.model_dir / 'faithfulness_model.pkl'
        }
        
        # Load scaler
        scaler_path = self.model_dir / 'feature_scaler.pkl'
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.debug(f"Loaded feature scaler from {scaler_path}")
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}")
        
        # Load feature names
        feature_names_path = self.model_dir / 'feature_names.pkl'
        if feature_names_path.exists():
            try:
                with open(feature_names_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                logger.debug(f"Loaded feature names: {self.feature_names}")
            except Exception as e:
                logger.warning(f"Failed to load feature names: {e}")
        
        # Load models
        for model_name, model_path in model_files.items():
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    self.models[model_name] = model
                    self.available_models.append(model_name)
                    logger.debug(f"Loaded {model_name} model from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} model: {e}")

    def is_available(self) -> bool:
        """Check if any trained models are available."""
        return len(self.available_models) > 0

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make predictions using trained models.
        
        Args:
            features: Dictionary with feature values:
                - semantic_relevance: float [0,1]
                - context_overlap: float [0,1]
                - confidence_score: float [0,1]
                - context_length: int
                - num_context_docs: int
        
        Returns:
            Dictionary with predictions:
            {
                "relevance": float,
                "hallucination": float,
                "faithfulness": float,
                "status": "success" or "partial"
            }
        """
        if not self.available_models:
            logger.warning("No trained models available for prediction")
            return {
                "relevance": None,
                "hallucination": None,
                "faithfulness": None,
                "status": "no_models"
            }
        
        try:
            # Prepare features in correct order
            feature_array = self._prepare_feature_array(features)
            
            # Scale features if scaler available
            if self.scaler is not None:
                feature_array = self.scaler.transform(feature_array.reshape(1, -1))
            else:
                feature_array = feature_array.reshape(1, -1)
            
            # Make predictions
            predictions = {
                "status": "success"
            }
            
            for model_name in ['relevance', 'hallucination', 'faithfulness']:
                if model_name in self.models:
                    try:
                        model = self.models[model_name]
                        
                        # Get probability prediction
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(feature_array)
                            # Probability of class 1
                            pred_value = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
                        else:
                            # For models without predict_proba
                            pred = model.predict(feature_array)[0]
                            pred_value = float(pred)
                        
                        predictions[model_name] = max(0.0, min(1.0, pred_value))  # Clamp to [0, 1]
                    except Exception as e:
                        logger.error(f"Prediction failed for {model_name}: {e}")
                        predictions[model_name] = None
                else:
                    predictions[model_name] = None
            
            # Update status if any predictions failed
            if any(v is None for k, v in predictions.items() if k != "status"):
                predictions["status"] = "partial"
            
            logger.debug(f"Predictions: {predictions}")
            return predictions
        
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            return {
                "relevance": None,
                "hallucination": None,
                "faithfulness": None,
                "status": "error"
            }

    def _prepare_feature_array(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare feature array in correct order with default values.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Numpy array of features
        """
        # Default feature names if not loaded
        default_features = [
            'semantic_relevance',
            'context_overlap',
            'confidence_score',
            'context_length',
            'num_context_docs'
        ]
        
        feature_list = self.feature_names if self.feature_names else default_features
        
        # Extract features in order, using default values where missing
        feature_array = []
        for fname in feature_list:
            value = features.get(fname, 0.5)  # Default value
            # Ensure numeric
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0.5
            feature_array.append(value)
        
        return np.array(feature_array, dtype=np.float32)
