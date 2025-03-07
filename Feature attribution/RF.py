import numpy as np
import logging
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import joblib

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

X = feature_all.drop(negative_infor.columns, axis=1)
y = feature_all[['Label']]

# Divide the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
results_df = pd.DataFrame(columns=["trial", "n_estimators", "max_depth", "min_samples_split",
                                   "min_samples_leaf", "max_features", "bootstrap", "score"])


# Define the objective function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 30)
    max_depth = trial.suggest_int('max_depth', 4, 32, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42
    )
    
    score = cross_val_score(model, X_train, y_train, cv=5).mean()  # 使用五折交叉验证计算平均得分
    results_df.loc[len(results_df)] = [
        trial.number,
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        score
    ]
    
    logger.info(f"Trial: {trial.number}, Parameters: n_estimators={n_estimators}, max_depth={max_depth}, "
                f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                f"max_features={max_features}, bootstrap={bootstrap}, Score: {score}")

    return score
