"""
ML scoring service with sklearn and SHAP explanations.
Replaces rules-based eligibility with ML-based decisions.

Changes:
- Robust handling of validation['flags'] (list/dict/other)
- Safer numeric parsing for all fields
- Keeps SHAP, but guards common edge cases
"""

import os
import json
import pickle
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import shap  # ensure installed

from backend.fastapi_app.ml.synthetic_data import generate_synthetic_dataset


def _to_float(x, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _to_int(x, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _normalize_flags(flags_obj) -> Dict[str, bool]:
    """
    Accepts:
      - dict like {"employment_consistent": True}
      - list like ["employment_consistent", "has_id_doc"]
      - anything else -> {}
    """
    if isinstance(flags_obj, dict):
        # coerce all truthy to bool
        return {str(k): bool(v) for k, v in flags_obj.items()}
    if isinstance(flags_obj, list):
        return {str(k): True for k in flags_obj}
    return {}


class SocialSupportMLModel:
    """ML model for social support eligibility scoring."""

    def __init__(self):
        self.model: Optional[GradientBoostingClassifier] = None
        self.explainer = None
        self.feature_names: List[str] = [
            "declared_monthly_income",
            "household_size",
            "extracted_monthly_income",
            "employment_months",
            "has_assets",
            "debt_to_income_ratio",
        ]
        self.is_trained: bool = False

    def extract_features(self, application_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from application data for prediction.

        Args:
            application_data: Dict containing validation results and app metadata

        Returns:
            Feature array ready for prediction (shape: (1, n_features))
        """
        validation = application_data.get("validation", {}) or {}
        app_row = application_data.get("app_row", {}) or {}

        # Parse numerics robustly
        declared_income = _to_float(app_row.get("declared_monthly_income"), 0.0)
        household_size = _to_int(app_row.get("household_size"), 1)

        # Income extracted from validation (proxy)
        extracted_income = _to_float(validation.get("total_inflow_30d"), declared_income)
        total_outflow_30d = _to_float(validation.get("total_outflow_30d"), 0.0)

        # Flags normalization
        flags = _normalize_flags(validation.get("flags"))
        employment_consistent = bool(flags.get("employment_consistent", True))

        # Employment months (simple heuristic)
        employment_months = 12.0 if employment_consistent else 3.0

        # Assets heuristic (keep your original logic)
        has_assets = 1 if total_outflow_30d > declared_income * 0.8 else 0

        # Debt ratio (avoid div by zero)
        if declared_income > 0:
            debt_ratio = min(0.9, total_outflow_30d / max(declared_income, 1.0))
        else:
            debt_ratio = 0.5  # neutral prior if no declared income

        features = np.array(
            [
                declared_income,
                household_size,
                extracted_income,
                employment_months,
                has_assets,
                debt_ratio,
            ],
            dtype=float,
        ).reshape(1, -1)

        return features

    def train_model(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Train the ML model on synthetic data."""
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            df = generate_synthetic_dataset(n_samples=2000)

        X = df[self.feature_names].values
        y = df["eligible"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        # SHAP explainer for tree models
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            # If SHAP fails (version mismatch or env issue), we still allow predictions.
            self.explainer = None

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        training_results = {
            "accuracy": float(report.get("accuracy", 0.0)),
            "precision": float(report.get("1", {}).get("precision", 0.0)),
            "recall": float(report.get("1", {}).get("recall", 0.0)),
            "f1_score": float(report.get("1", {}).get("f1-score", 0.0)),
            "confusion_matrix": cm.tolist(),
            "feature_importance": {
                name: float(importance)
                for name, importance in zip(
                    self.feature_names, getattr(self.model, "feature_importances_", [0] * len(self.feature_names))
                )
            },
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        }

        self.is_trained = True
        return training_results

    def predict(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with SHAP explanations.

        Args:
            application_data: Application data with validation results

        Returns:
            Dict with prediction, probability, and SHAP explanations
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        features = self.extract_features(application_data)

        prediction = int(self.model.predict(features)[0])
        prediction_proba = self.model.predict_proba(features)[0]
        approve_prob = float(prediction_proba[1])
        decline_prob = float(prediction_proba[0])

        # SHAP explanations (guarded)
        shap_explanation = {}
        top_factors = []
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(features)
                # Older TreeExplainer returns array for binary models; some versions return list
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                shap_row = shap_values[0]

                for i, fname in enumerate(self.feature_names):
                    sv = float(shap_row[i])
                    val = float(features[0, i])
                    shap_explanation[fname] = {
                        "value": val,
                        "shap_value": sv,
                        "contribution": "positive" if sv > 0 else "negative",
                    }

                # sort by absolute impact
                sorted_items = sorted(
                    shap_explanation.items(),
                    key=lambda kv: abs(kv[1]["shap_value"]),
                    reverse=True,
                )
                top_factors = [
                    {
                        "feature": name,
                        "value": data["value"],
                        "impact": data["shap_value"],
                        "direction": data["contribution"],
                    }
                    for name, data in sorted_items[:3]
                ]
            except Exception:
                shap_explanation = {}
                top_factors = []

        return {
            "prediction": prediction,
            "decision": "APPROVE" if prediction == 1 else "DECLINE",
            "confidence": float(max(approve_prob, decline_prob)),
            "approve_probability": approve_prob,
            "decline_probability": decline_prob,
            "shap_explanation": shap_explanation,
            "top_factors": top_factors,
        }

    def save_model(self, model_dir: str = "data/ml") -> str:
        """Save trained model to disk."""
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "social_support_model.pkl")

        model_data = {
            "model": self.model,
            "explainer": self.explainer,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return model_path

    def load_model(self, model_path: str) -> bool:
        """Load trained model from disk."""
        if not os.path.exists(model_path):
            return False
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data.get("model")
            self.explainer = model_data.get("explainer")
            self.feature_names = model_data.get("feature_names", self.feature_names)
            self.is_trained = bool(model_data.get("is_trained", False))
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# Global model instance
_model_instance: Optional[SocialSupportMLModel] = None


def get_ml_model() -> SocialSupportMLModel:
    """Get or create the global ML model instance."""
    global _model_instance

    if _model_instance is None:
        _model_instance = SocialSupportMLModel()
        model_path = "data/ml/social_support_model.pkl"
        if not _model_instance.load_model(model_path):
            print("Training new ML model...")
            training_results = _model_instance.train_model()
            _model_instance.save_model()
            print(f"Model trained with accuracy: {training_results.get('accuracy', 0.0):.3f}")

    return _model_instance


def score_application(application_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score an application using the ML model.
    """
    model = get_ml_model()
    return model.predict(application_data)


if __name__ == "__main__":
    # Quick local sanity
    model = SocialSupportMLModel()
    results = model.train_model()
    print("Training Results:", json.dumps(results, indent=2))

    test_data = {
        "app_row": {"declared_monthly_income": 2500, "household_size": 3},
        "validation": {
            "total_inflow_30d": 2400,
            "total_outflow_30d": 2000,
            "flags": ["employment_consistent"],  # list form is now supported
        },
    }

    prediction = model.predict(test_data)
    print("Test Prediction:", json.dumps(prediction, indent=2))
