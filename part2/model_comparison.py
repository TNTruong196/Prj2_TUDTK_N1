"""
So sanh mo hinh OLS va Ridge tren du lieu Air Quality da tien xu ly.

Module nay dung cac ham tu cai dat o part1 cho phan fit/predict chinh.
Sklearn chi duoc dung de chia train/test reproducible.
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part1.cross_validation import kfold_cv
from part1.matrix_helper import mat_mul
from part1.ols_implementation import ols_fit
from part1.ridge_lasso import plot_ridge_trace, ridge_fit

try:
    from part2.data_pipeline import DataPipeline
except ModuleNotFoundError:
    from data_pipeline import DataPipeline


SEED = 42
TARGET_COL = "C6H6(GT)"
DATA_PATH = PROJECT_ROOT / "part2" / "data" / "AirQuality.csv"


def default_lambdas():
    """Tao grid lambda tu 10^-4 den 10^4."""
    return [10 ** (-4 + 8 * i / 24) for i in range(25)]


def df_to_2d_list(data):
    """Chuyen pandas DataFrame/Series hoac list 1D thanh 2D list."""
    if isinstance(data, pd.Series):
        return [[float(value)] for value in data.tolist()]

    if isinstance(data, pd.DataFrame):
        return data.astype(float).values.tolist()

    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], list):
            return data
        return [[float(value)] for value in data]

    raise TypeError("data phai la pandas DataFrame, pandas Series hoac list")


def add_intercept(X):
    """Them cot intercept 1.0 vao dau ma tran X."""
    return [[1.0] + [float(value) for value in row] for row in X]


def predict(X, beta):
    """Du doan y_hat = X @ beta."""
    return mat_mul(X, beta)


def compute_metrics(y_true, y_pred):
    """Tinh MAE, RMSE va R2 cho bai toan hoi quy."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true va y_pred phai co cung so dong")
    if len(y_true) == 0:
        raise ValueError("Khong the tinh metrics tren tap rong")

    y_values = [row[0] for row in y_true]
    pred_values = [row[0] for row in y_pred]
    n = len(y_values)

    abs_errors = [abs(y_values[i] - pred_values[i]) for i in range(n)]
    sq_errors = [(y_values[i] - pred_values[i]) ** 2 for i in range(n)]

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)

    y_bar = sum(y_values) / n
    rss = sum(sq_errors)
    tss = sum((value - y_bar) ** 2 for value in y_values)
    r2 = 1 - rss / tss if tss != 0 else 0.0

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def select_best_lambda(X, y, k=5, lambdas=None):
    """
    Chon lambda tot nhat cho Ridge bang k-fold CV.

    Returns:
        best_lambda: lambda co CV-MSE nho nhat
        cv_results: DataFrame gom lambda, log10_lambda, cv_mse, fold_mses
    """
    if lambdas is None:
        lambdas = default_lambdas()
    if not lambdas:
        raise ValueError("lambdas khong duoc rong")

    records = []
    for lam in lambdas:
        if lam < 0:
            raise ValueError("lambda phai >= 0")

        fit_func = lambda X_train, y_train, current_lam=lam: ridge_fit(
            X_train, y_train, current_lam
        )
        cv_mse, fold_mses = kfold_cv(X, y, k, fit_func=fit_func)
        records.append(
            {
                "lambda": lam,
                "log10_lambda": math.log10(lam) if lam > 0 else None,
                "cv_mse": cv_mse,
                "fold_mses": fold_mses,
            }
        )

    cv_results = pd.DataFrame(records)
    best_row = cv_results.loc[cv_results["cv_mse"].idxmin()]
    return float(best_row["lambda"]), cv_results


def plot_cv_results(cv_results, show=True):
    """Ve CV-MSE theo log10(lambda)."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        cv_results["log10_lambda"],
        cv_results["cv_mse"],
        marker="o",
        linewidth=1.5,
    )
    plt.title("Ridge Lambda Selection by K-Fold CV")
    plt.xlabel("log10(lambda)")
    plt.ylabel("CV-MSE")
    plt.grid(True)
    if show:
        plt.show()


def ridge_coefficients_table(beta, feature_names):
    """Tao bang he so Ridge/OLS kem ten feature."""
    records = []
    for i, row in enumerate(beta):
        name = feature_names[i] if i < len(feature_names) else f"beta_{i}"
        records.append({"Feature": name, "Coefficient": row[0]})

    table = pd.DataFrame(records)
    table["AbsCoefficient"] = table["Coefficient"].abs()
    return table.sort_values("AbsCoefficient", ascending=False).drop(
        columns=["AbsCoefficient"]
    )


def ridge_trace_table(X, y, lambdas, feature_names=None, decimals=4, max_features=10):
    """
    Tao bang Ridge Trace de doc thay cho list he so tho.

    Moi dong la mot lambda, moi cot la he so cua mot feature.
    Mac dinh chi hien thi 10 he so feature dau tien de bang khong qua rong.
    """
    if not lambdas:
        raise ValueError("lambdas khong duoc rong")

    beta_traces = []
    for lam in lambdas:
        beta = ridge_fit(X, y, lam)
        beta_traces.append([row[0] for row in beta])

    n_coef = len(beta_traces[0]) if beta_traces else 0
    names = feature_names or [f"beta_{i}" for i in range(n_coef)]

    coef_indices = list(range(n_coef))
    if max_features is not None:
        start_idx = 1 if n_coef > 1 else 0
        end_idx = min(n_coef, start_idx + max_features)
        coef_indices = [0] + list(range(start_idx, end_idx)) if n_coef > 1 else [0]

    records = []
    for lam, beta in zip(lambdas, beta_traces):
        row = {
            "lambda": lam,
            "log10(lambda)": math.log10(lam) if lam > 0 else None,
        }
        for idx in coef_indices:
            name = names[idx] if idx < len(names) else f"beta_{idx}"
            row[name] = beta[idx]
        records.append(row)

    table = pd.DataFrame(records)
    return table.round(decimals)


def prepare_air_quality_data(data_path=DATA_PATH, test_size=0.2):
    """Load data, tach target, chia train/test va chay DataPipeline."""
    df = pd.read_csv(data_path)
    df_clean = df.dropna(subset=[TARGET_COL]).copy()

    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SEED,
    )

    pipeline = DataPipeline(missing_threshold=0.5, numeric_strategy="median")
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    X_train_2d = add_intercept(df_to_2d_list(X_train_processed))
    X_test_2d = add_intercept(df_to_2d_list(X_test_processed))
    y_train_2d = df_to_2d_list(y_train)
    y_test_2d = df_to_2d_list(y_test)

    return {
        "pipeline": pipeline,
        "feature_names": ["intercept"] + X_train_processed.columns.tolist(),
        "X_train": X_train_2d,
        "X_test": X_test_2d,
        "y_train": y_train_2d,
        "y_test": y_test_2d,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
    }


def train_and_compare(data_path=DATA_PATH, k=5, lambdas=None, plot=False):
    """
    Train OLS co ban va Ridge(lambda tot nhat), sau do danh gia tren test set.

    Returns:
        dict gom best_lambda, cv_results, metrics_table, models, data
    """
    data = prepare_air_quality_data(data_path=data_path)

    best_lambda, cv_results = select_best_lambda(
        data["X_train"],
        data["y_train"],
        k=k,
        lambdas=lambdas,
    )

    beta_ols, sigma2_ols = ols_fit(data["X_train"], data["y_train"])
    beta_ridge = ridge_fit(data["X_train"], data["y_train"], best_lambda)

    y_pred_ols = predict(data["X_test"], beta_ols)
    y_pred_ridge = predict(data["X_test"], beta_ridge)

    metrics_table = pd.DataFrame(
        [
            {"Model": "OLS", **compute_metrics(data["y_test"], y_pred_ols)},
            {
                "Model": f"Ridge (lambda={best_lambda:.6g})",
                **compute_metrics(data["y_test"], y_pred_ridge),
            },
        ]
    )

    if plot:
        plot_cv_results(cv_results)
        plot_ridge_trace(data["X_train"], data["y_train"], lambdas or default_lambdas())

    return {
        "best_lambda": best_lambda,
        "cv_results": cv_results,
        "metrics_table": metrics_table,
        "models": {
            "ols": {"beta": beta_ols, "sigma2": sigma2_ols},
            "ridge": {"beta": beta_ridge, "lambda": best_lambda},
        },
        "predictions": {
            "ols": y_pred_ols,
            "ridge": y_pred_ridge,
        },
        "data": data,
    }


def main():
    result = train_and_compare(k=5, plot=False)

    print(f"Best lambda: {result['best_lambda']:.6g}")
    print("\nTop 5 lambda theo CV-MSE:")
    print(
        result["cv_results"]
        .sort_values("cv_mse")[["lambda", "cv_mse"]]
        .head()
        .to_string(index=False)
    )
    print("\nTest metrics:")
    print(result["metrics_table"].to_string(index=False))


if __name__ == "__main__":
    main()
