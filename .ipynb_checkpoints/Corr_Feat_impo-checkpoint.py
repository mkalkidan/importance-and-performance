import os
import numpy as np
import pandas as pd
import scipy.io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import shap
from lime.lime_tabular import LimeTabularExplainer

import shap
import numpy as np
import pandas as pd

import warnings

# Use the built-in Python UserWarning directly
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

def lr_shap(model, X_train, X_sample):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_sample, check_additivity=False)

    shap_vals = shap_values.values

    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]

    importance = np.abs(shap_vals).mean(axis=0)

    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": importance
    }).sort_values("importance", ascending=False)

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd

def lr_lime(model, X_train, X_test, num_samples=50):

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=["0", "1"],
        mode="classification"
    )

    importances = np.zeros(X_train.shape[1])
    actual_samples = min(num_samples, len(X_test))

    idxs = np.random.choice(len(X_test), actual_samples, replace=False)

    for i in idxs:
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba
        )

        for feature, weight in exp.as_list():
            feature_name = feature.split()[0]

            if feature_name in X_train.columns:
                idx = X_train.columns.get_loc(feature_name)
                importances[idx] += abs(weight)

    importances = importances / actual_samples

    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)


def rf_shap(model, X_train, X_sample):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_sample, check_additivity=False)

    shap_vals = shap_values.values

    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]

    importance = np.abs(shap_vals).mean(axis=0)

    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": importance
    }).sort_values("importance", ascending=False)


def rf_lime(model, X_train, X_test, num_samples=50):

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=["0", "1"],
        mode="classification"
    )

    importances = np.zeros(X_train.shape[1])
    actual_samples = min(num_samples, len(X_test))

    idxs = np.random.choice(len(X_test), actual_samples, replace=False)

    for i in idxs:
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict_proba
        )

        for feature, weight in exp.as_list():
            feature_name = feature.split()[0]

            if feature_name in X_train.columns:
                idx = X_train.columns.get_loc(feature_name)
                importances[idx] += abs(weight)

    importances = importances / actual_samples

    return pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)