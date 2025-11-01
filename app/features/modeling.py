from typing import Callable, Optional, Tuple
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def run_classification(
    df: pd.DataFrame,
    cv_k: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Train and compare KNN, RandomForest, SVM with GridSearchCV using F1-macro.

    Returns a sorted DataFrame with columns: Model, F1 Score (Macro), Best Hyperparameters.
    Optionally reports progress via progress_callback(i, total).
    """
    X = df.drop(columns=["Id", "Type"])
    y = df["Type"]

    f1 = make_scorer(f1_score, average="macro")
    kfold = KFold(n_splits=cv_k, shuffle=True, random_state=42)

    models = {
        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
            {"knn__n_neighbors": [3, 5, 7, 9], "knn__weights": ["uniform", "distance"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        ),
        "SVM (RBF)": (
            Pipeline([("scaler", StandardScaler()), ("svm", SVC())]),
            {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", "auto"], "svm__kernel": ["rbf"]},
        ),
    }

    rows = []
    total = len(models)
    for i, (name, (model, params)) in enumerate(models.items(), start=1):
        grid = GridSearchCV(model, params, cv=kfold, scoring=f1, n_jobs=-1)
        grid.fit(X, y)
        rows.append([name, round(grid.best_score_, 4), grid.best_params_])
        if progress_callback:
            progress_callback(i, total)

    results = pd.DataFrame(
        rows,
        columns=["Model", "F1 Score (Macro)", "Best Hyperparameters"],
    ).sort_values("F1 Score (Macro)", ascending=False)

    return results
