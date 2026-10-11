"""
model_inference.py  --  Person 2, Model serialization & inference engine

Provides:
- Serialization and loading of the trained pipeline using joblib
- Packaging model metadata (feature schema, classes, training period, performance)
- EPLMatchPredictor inference class with:
    * Input schema validation
    * Column order alignment
    * Missing value safety
    * Prediction of outcome (H, D, A) and class probabilities
    * Explanation of predictions
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.model_evaluation import CLASS_LABELS_FULL, CLASS_ORDER


class EPLMatchPredictor:
    """Production inference interface for the serialized EPL match outcome model."""

    def __init__(self, pipeline: Pipeline, metadata: Dict[str, Any]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.feature_columns: List[str] = metadata["feature_columns"]
        self.classes: List[str] = metadata.get("classes", CLASS_ORDER)
        self.model_name: str = metadata.get("model_name", "Unknown")
        self.feature_group: str = metadata.get("feature_group", "Unknown")

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        metadata_path: Union[str, Path],
    ) -> EPLMatchPredictor:
        """Load serialized model and metadata from disk."""
        pipeline = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(pipeline=pipeline, metadata=metadata)

    def save(
        self,
        model_path: Union[str, Path],
        metadata_path: Union[str, Path],
    ) -> None:
        """Save model and metadata to disk."""
        m_path = Path(model_path)
        meta_path = Path(metadata_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, m_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def validate_and_align_features(
        self,
        X_input: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Validate input features against required schema and align column ordering."""
        if isinstance(X_input, dict):
            df = pd.DataFrame([X_input])
        elif isinstance(X_input, list):
            df = pd.DataFrame(X_input)
        elif isinstance(X_input, pd.DataFrame):
            df = X_input.copy()
        else:
            raise TypeError(f"Unsupported input type for features: {type(X_input)}")

        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(
                f"Input data is missing {len(missing_features)} required feature(s): {missing_features[:5]}..."
            )

        # Align column order strictly to training expectations
        return df[self.feature_columns]

    def predict(
        self,
        X_input: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Generate outcome predictions and probabilities for input fixture features."""
        X_aligned = self.validate_and_align_features(X_input)
        preds = self.pipeline.predict(X_aligned)

        results = pd.DataFrame({
            "predicted_outcome": preds,
            "predicted_label_full": [CLASS_LABELS_FULL.get(p, p) for p in preds],
        })

        # Calculate probabilities if supported
        if hasattr(self.pipeline, "predict_proba"):
            try:
                probs = self.pipeline.predict_proba(X_aligned)
                # Map probabilities to classes
                classes_in_model = list(self.pipeline.classes_)
                for cls_name in CLASS_ORDER:
                    if cls_name in classes_in_model:
                        idx = classes_in_model.index(cls_name)
                        results[f"prob_{cls_name}"] = probs[:, idx].round(4)
                    else:
                        results[f"prob_{cls_name}"] = 0.0
            except Exception:
                pass

        return results

    def predict_single(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method for predicting a single match fixture."""
        df_res = self.predict(feature_dict)
        rec = df_res.iloc[0].to_dict()
        return rec


def save_trained_model(
    pipeline: Pipeline,
    feature_columns: List[str],
    model_name: str,
    feature_group: str,
    training_period: str,
    validation_period: str,
    test_period: str,
    hyperparameters: Dict[str, Any],
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Serialize the final model pipeline and save comprehensive metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "epl_model.joblib"
    metadata_file = output_dir / "model_metadata.json"

    classes_list = [str(c) for c in pipeline.classes_]

    metadata = {
        "model_name": model_name,
        "feature_group": feature_group,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "classes": classes_list,
        "training_period": training_period,
        "validation_period": validation_period,
        "test_period": test_period,
        "selected_hyperparameters": hyperparameters,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "serialization_format": "joblib",
    }

    predictor = EPLMatchPredictor(pipeline=pipeline, metadata=metadata)
    predictor.save(model_file, metadata_file)

    return model_file, metadata_file
