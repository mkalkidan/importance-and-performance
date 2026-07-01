import os
import scipy.io
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.io
import os
import numpy as np
from scipy.stats import pearsonr
import lime 
import lime.lime_tabular
import numpy as np 
import shap

import pandas as pd
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def load_dataset(dataset_number):
    # Base directory
    base_dir = os.path.join(
        os.path.expanduser("~"),
        "Desktop",
        "TEAM - Dataset",
        "data"
    )
    
    # Build full path dynamically
    file_path = os.path.join(base_dir, str(dataset_number), "data.mat")
    
    # Load .mat file
    data = scipy.io.loadmat(file_path)
    
    # Extract variables
    X, y = data['X'], data['y']
    
    # Convert X to DataFrame with feature names
    X = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(X.shape[1])]
    )
    
    print(f"Dataset {dataset_number} loaded.")
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    return X, y
######################################################################

def train_test_split_custom(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.ravel(),
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test
######################################################################
def train_rf_model(X_train, y_train):
    RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
    RF_model.fit(X_train, y_train)
    return RF_model
####################################################################
def train_lr_model(X_train, y_train):
    LR_model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=42
    )
    LR_model.fit(X_train, y_train)
    return LR_model
######################################################################
from sklearn.metrics import roc_auc_score

def evaluate_auc(model, X, y):
    # Both Logistic Regression and Random Forest support predict_proba
    y_scores = model.predict_proba(X)[:, 1]
    
    auc = roc_auc_score(y, y_scores)
    return round(auc, 4)