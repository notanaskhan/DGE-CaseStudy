"""
Synthetic data generation for social support ML model.
Simple features focused on core eligibility factors.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import os
import json


def generate_synthetic_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dataset with simple, interpretable features.

    Features:
    1. declared_monthly_income - What applicant declares
    2. household_size - Number of people in household
    3. extracted_monthly_income - Income from bank statements/docs
    4. employment_months - Months of employment history
    5. has_assets - Whether applicant has significant assets
    6. debt_to_income_ratio - Total debt vs income ratio

    Target: eligible (1 = approve, 0 = decline)
    """
    np.random.seed(random_state)

    data = []

    for i in range(n_samples):
        # Generate base income (realistic UAE range)
        true_income = np.random.lognormal(mean=8.0, sigma=0.6)  # 2000-8000 AED typical range
        true_income = max(1000, min(15000, true_income))  # Cap between 1k-15k AED

        # Household size (1-8 people, weighted toward smaller)
        household_size = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],
                                        p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02])

        # Declared income - sometimes differs from true income
        income_accuracy = np.random.uniform(0.8, 1.2)  # ±20% variation
        declared_income = true_income * income_accuracy

        # Extracted income from documents - closer to true but with noise
        extraction_noise = np.random.uniform(0.9, 1.1)  # ±10% extraction accuracy
        extracted_income = true_income * extraction_noise

        # Employment history (0-60 months)
        employment_months = np.random.exponential(scale=20)  # Average 20 months
        employment_months = min(60, max(0, employment_months))

        # Assets - binary indicator
        has_assets = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% have significant assets

        # Debt to income ratio
        if has_assets:
            debt_ratio = np.random.uniform(0.1, 0.4)  # Lower debt if assets
        else:
            debt_ratio = np.random.uniform(0.2, 0.8)  # Higher debt if no assets

        # Calculate per-capita income
        per_capita_income = extracted_income / household_size

        # Eligibility logic (simplified but realistic)
        # Base threshold: 3000 AED per capita
        # Adjustments for employment stability and debt
        base_threshold = 3000

        # Employment stability factor
        if employment_months >= 12:
            emp_factor = 1.1  # 10% bonus for stable employment
        elif employment_months >= 6:
            emp_factor = 1.0
        else:
            emp_factor = 0.9  # 10% penalty for unstable employment

        # Debt factor
        if debt_ratio > 0.6:
            debt_factor = 0.9  # Penalty for high debt
        elif debt_ratio < 0.3:
            debt_factor = 1.05  # Small bonus for low debt
        else:
            debt_factor = 1.0

        adjusted_threshold = base_threshold / (emp_factor * debt_factor)

        # Final eligibility decision
        eligible = 1 if per_capita_income <= adjusted_threshold else 0

        # Add some random noise to make it more realistic (5% random flips)
        if np.random.random() < 0.05:
            eligible = 1 - eligible

        data.append({
            'declared_monthly_income': round(declared_income, 2),
            'household_size': household_size,
            'extracted_monthly_income': round(extracted_income, 2),
            'employment_months': round(employment_months, 1),
            'has_assets': has_assets,
            'debt_to_income_ratio': round(debt_ratio, 3),
            'eligible': eligible
        })

    return pd.DataFrame(data)


def save_synthetic_dataset(data_dir: str = "data/ml") -> str:
    """Save synthetic dataset to data directory."""
    os.makedirs(data_dir, exist_ok=True)

    # Generate dataset
    df = generate_synthetic_dataset()

    # Save as CSV
    dataset_path = os.path.join(data_dir, "synthetic_training_data.csv")
    df.to_csv(dataset_path, index=False)

    # Save metadata
    metadata = {
        "n_samples": len(df),
        "features": list(df.columns),
        "target": "eligible",
        "feature_descriptions": {
            "declared_monthly_income": "Monthly income declared by applicant (AED)",
            "household_size": "Number of people in household",
            "extracted_monthly_income": "Monthly income extracted from documents (AED)",
            "employment_months": "Months of employment history",
            "has_assets": "Binary indicator for significant assets (1=yes, 0=no)",
            "debt_to_income_ratio": "Total debt divided by monthly income",
            "eligible": "Target variable (1=approve, 0=decline)"
        },
        "statistics": {
            "approval_rate": float(df['eligible'].mean()),
            "avg_declared_income": float(df['declared_monthly_income'].mean()),
            "avg_household_size": float(df['household_size'].mean())
        }
    }

    metadata_path = os.path.join(data_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Synthetic dataset saved to: {dataset_path}")
    print(f"Dataset metadata saved to: {metadata_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Approval rate: {df['eligible'].mean():.2%}")

    return dataset_path


if __name__ == "__main__":
    save_synthetic_dataset()