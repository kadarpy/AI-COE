"""
MLflow Experiment Tracking
===========================
Tracks evaluation metrics, parameters, and artifacts for reproducibility.

Features:
- Experiment-based organization
- Metric logging
- Parameter tracking
- Artifact storage (training data, models)
- Run management
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Manages MLflow experiment tracking for RAG evaluation system.
    Supports local file-based tracking with mlruns/ directory.
    """

    def __init__(self, tracking_uri: str = None, experiment_name: str = "rag_evaluation"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: Path to MLflow tracking directory (defaults to ./mlruns)
            experiment_name: Name of MLflow experiment
        """
        # Set tracking URI (default to local mlruns directory)
        if tracking_uri is None:
            project_root = Path(__file__).parent.parent.parent
            tracking_uri = str(project_root / "mlruns")
        
        # Create tracking directory if it doesn't exist
        Path(tracking_uri).mkdir(parents=True, exist_ok=True)
        
        # Configure MLflow
        mlflow.set_tracking_uri(f"file:{tracking_uri}")
        self.client = MlflowClient(tracking_uri=f"file:{tracking_uri}")
        self.tracking_uri = tracking_uri
        
        # Set or get experiment
        try:
            self.experiment = self.client.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                exp_id = mlflow.create_experiment(experiment_name)
                self.experiment = self.client.get_experiment(exp_id)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment: {e}")
            raise
        
        self.experiment_name = experiment_name
        self.current_run_id = None
        
        logger.info(f"MLflow tracker initialized: {experiment_name} at {tracking_uri}")

    def start_run(self, run_name: str, tags: Dict[str, str] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Optional dictionary of tags
            
        Returns:
            Run ID
        """
        try:
            mlflow.start_run(run_name=run_name)
            run = mlflow.active_run()
            self.current_run_id = run.info.run_id
            
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, str(value))
            
            logger.info(f"Started MLflow run: {run_name} (ID: {self.current_run_id})")
            return self.current_run_id
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        if not mlflow.active_run():
            logger.warning("No active MLflow run. Metrics not logged.")
            return
        
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
                else:
                    logger.warning(f"Skipping metric '{key}': non-numeric value {value}")
            
            logger.debug(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if not mlflow.active_run():
            logger.warning("No active MLflow run. Parameters not logged.")
            return
        
        try:
            for key, value in params.items():
                mlflow.log_param(key, str(value))
            
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_artifact(self, file_path: str, artifact_path: str = None) -> None:
        """
        Log an artifact file to MLflow.
        
        Args:
            file_path: Path to file to log
            artifact_path: Optional subdirectory for artifact
        """
        if not mlflow.active_run():
            logger.warning("No active MLflow run. Artifact not logged.")
            return
        
        try:
            if not Path(file_path).exists():
                logger.error(f"Artifact file not found: {file_path}")
                return
            
            mlflow.log_artifact(file_path, artifact_path=artifact_path)
            logger.debug(f"Logged artifact: {file_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        try:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info(f"Ended MLflow run: {self.current_run_id}")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def get_experiment_info(self) -> Dict[str, Any]:
        """Get information about the current experiment."""
        return {
            "experiment_id": self.experiment.experiment_id,
            "experiment_name": self.experiment.name,
            "tracking_uri": self.tracking_uri
        }


# Global tracker instance (lazy initialization)
_tracker_instance: Optional[MLflowTracker] = None


def get_mlflow_tracker(tracking_uri: str = None) -> MLflowTracker:
    """
    Get or create MLflow tracker instance.
    
    Args:
        tracking_uri: Optional path to MLflow tracking directory
        
    Returns:
        MLflowTracker instance
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = MLflowTracker(tracking_uri=tracking_uri)
    return _tracker_instance
