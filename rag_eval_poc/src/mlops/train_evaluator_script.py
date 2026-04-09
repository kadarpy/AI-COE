#!/usr/bin/env python3
"""
Model Training Script for Evaluator
====================================
Trains XGBoost/LogisticRegression models using evaluation data.

Usage:
    python src/mlops/train_evaluator_script.py

Requirements:
    - training_data.jsonl with evaluation records (from batch evaluations)
    - python-xgboost, scikit-learn installed
"""

import sys
import logging
from pathlib import Path

# Add parent to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.mlops.train_evaluator import EvaluatorModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training entry point"""
    logger.info("=" * 70)
    logger.info("RAG EVALUATOR MODEL TRAINING")
    logger.info("=" * 70)
    
    # Check if training data exists
    training_data_path = Path(config.TRAINING_DATA_PATH)
    if not training_data_path.exists():
        logger.error(f"❌ Training data not found: {training_data_path}")
        logger.error("Run batch evaluations first to generate training data")
        return False
    
    logger.info(f"✓ Training data found: {training_data_path}")
    
    # Initialize trainer
    try:
        trainer = EvaluatorModelTrainer(
            training_data_path=str(training_data_path),
            model_dir=str(config.MODEL_DIR)
        )
        logger.info(f"✓ Trainer initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize trainer: {e}")
        return False
    
    # Train models
    try:
        results = trainer.train_models(test_size=0.2, random_state=42)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)
        
        for model_name, result in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Status: {result.get('status', 'unknown')}")
            logger.info(f"  Accuracy: {result.get('accuracy', 'N/A'):.4f}")
            logger.info(f"  F1 Score: {result.get('f1_score', 'N/A'):.4f}")
            logger.info(f"  Samples: {result.get('samples', 'N/A')}")
            logger.info(f"  Model Path: {result.get('model_path', 'N/A')}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nModels saved to: {config.MODEL_DIR}")
        logger.info("Next steps:")
        logger.info("1. Run batch evaluations in the Streamlit app")
        logger.info("2. View trained model predictions in 'Advanced Evaluation' tab")
        logger.info("3. Monitor MLflow runs: mlflow ui --backend-store-uri file:mlruns")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
