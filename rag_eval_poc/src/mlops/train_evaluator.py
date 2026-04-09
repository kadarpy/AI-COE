"""
Model Training Pipeline for ML Evaluator
==========================================
Trains supervised learning models using training data collected from evaluations.

Models trained:
- Relevance Model (XGBoost/LogisticRegression)
- Hallucination Model (XGBoost/LogisticRegression)
- Faithfulness Model (XGBoost/LogisticRegression)

Features extracted from evaluation records:
- semantic_relevance: CrossEncoder relevance score
- context_overlap: Token overlap between answer and context
- confidence_score: Combined metric
- context_length: Number of context documents
- num_context_docs: Count of retrieved documents
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Try XGBoost first, fall back to LogisticRegression
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class EvaluatorModelTrainer:
    """
    Trains ML models for predicting evaluation metrics.
    Supports XGBoost (preferred) or LogisticRegression (fallback).
    """

    def __init__(self, training_data_path: str, model_dir: str = "models"):
        """
        Initialize trainer.
        
        Args:
            training_data_path: Path to JSONL training data
            model_dir: Directory to save trained models
        """
        self.training_data_path = Path(training_data_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
        logger.info(f"EvaluatorModelTrainer initialized")
        logger.info(f"Training data: {self.training_data_path}")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Using XGBoost: {HAS_XGBOOST}")

    def load_training_data(self) -> pd.DataFrame:
        """
        Load training data from JSONL file.
        
        Returns:
            DataFrame with training data
            
        Raises:
            FileNotFoundError: If training data doesn't exist
            ValueError: If data is invalid
        """
        if not self.training_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.training_data_path}")
        
        records = []
        try:
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
            
            if not records:
                raise ValueError("No valid records found in training data")
            
            df = pd.DataFrame(records)
            logger.info(f"Loaded {len(df)} training records from {self.training_data_path}")
            
            return df
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract and prepare features from training data.
        
        Expected columns:
        - semantic_relevance: float [0,1]
        - context_overlap: float [0,1]
        - confidence_score: float [0,1]
        - context_length: int
        - num_context_docs: int
        
        Args:
            df: Training DataFrame
            
        Returns:
            Tuple of (features_array, feature_names)
        """
        feature_cols = [
            'semantic_relevance',
            'context_overlap',
            'confidence_score',
            'context_length',
            'num_context_docs'
        ]
        
        # Check which features are available
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            logger.warning("No feature columns found. Using default features.")
            # Fallback: create synthetic features from available data
            if 'semantic_relevance' not in df.columns:
                df['semantic_relevance'] = np.random.random(len(df))
            if 'context_overlap' not in df.columns:
                df['context_overlap'] = np.random.random(len(df))
            if 'confidence_score' not in df.columns:
                df['confidence_score'] = 0.5
            if 'context_length' not in df.columns:
                df['context_length'] = 1
            if 'num_context_docs' not in df.columns:
                df['num_context_docs'] = 1
            available_cols = feature_cols
        
        # Fill missing values
        for col in available_cols:
            if col in df.columns and df[col].isna().any():
                if col in ['context_length', 'num_context_docs']:
                    df[col] = df[col].fillna(1)  # Default to 1 for counts
                else:
                    df[col] = df[col].fillna(0.5)  # Default to 0.5 for probabilities
        
        # Extract features
        X = df[available_cols].values.astype(np.float32)
        
        logger.info(f"Extracted features: {available_cols}")
        logger.info(f"Feature shape: {X.shape}")
        
        return X, available_cols

    def train_models(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict]:
        """
        Train all three evaluator models.
        
        Args:
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results for each model
        """
        logger.info("=" * 70)
        logger.info("TRAINING EVALUATOR MODELS")
        logger.info("=" * 70)
        
        # Load data
        df = self.load_training_data()
        
        # Prepare features
        X, feature_names = self.prepare_features(df)
        self.feature_names = feature_names
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        
        # Train three models
        model_configs = {
            'relevance': {
                'label_col': 'labels.relevance' if 'labels.relevance' in df.columns else 'relevance',
                'output_file': self.model_dir / 'relevance_model.pkl'
            },
            'hallucination': {
                'label_col': 'labels.hallucination' if 'labels.hallucination' in df.columns else 'hallucination',
                'output_file': self.model_dir / 'hallucination_model.pkl'
            },
            'faithfulness': {
                'label_col': 'labels.faithfulness' if 'labels.faithfulness' in df.columns else 'faithfulness',
                'output_file': self.model_dir / 'faithfulness_model.pkl'
            }
        }
        
        results = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"\nTraining {model_name} model...")
            
            label_col = config['label_col']
            
            # Handle nested labels (labels.xxx)
            if '.' in label_col and label_col not in df.columns:
                parts = label_col.split('.')
                if parts[0] in df.columns:
                    df[label_col] = df[parts[0]].apply(
                        lambda x: x.get(parts[1]) if isinstance(x, dict) else None
                    )
            
            # Skip if labels not available
            if label_col not in df.columns:
                logger.warning(f"Label '{label_col}' not found. Skipping {model_name} model.")
                continue
            
            # Get labels
            y = df[label_col].values
            
            # Drop NaN labels
            valid_idx = ~pd.isna(y)
            X_valid = X_scaled[valid_idx]
            y_valid = y[valid_idx].astype(int)
            
            if len(y_valid) == 0:
                logger.warning(f"No valid labels for {model_name}. Skipping.")
                continue
            
            logger.info(f"Training samples: {len(y_valid)}")
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, test_size=test_size, random_state=random_state
            )
            
            # Train model
            if HAS_XGBOOST:
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=random_state,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    verbosity=0
                )
            else:
                model = LogisticRegression(random_state=random_state, max_iter=1000)
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
            recall = recall_score(y_test, y_pred, zero_division=0, average='weighted')
            f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")
            
            # Save model
            with open(config['output_file'], 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved: {config['output_file']}")
            
            # Store model
            self.models[model_name] = model
            
            results[model_name] = {
                'status': 'success',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'samples': len(y_valid),
                'model_path': str(config['output_file'])
            }
        
        # Save scaler
        scaler_path = self.model_dir / 'feature_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Feature scaler saved: {scaler_path}")
        
        # Save feature names
        feature_names_path = self.model_dir / 'feature_names.pkl'
        with open(feature_names_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        logger.info(f"Feature names saved: {feature_names_path}")
        
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        
        return results
