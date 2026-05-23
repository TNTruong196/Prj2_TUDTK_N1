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
from part1.ols_implementation import ols_fit, model_metrics, coef_inference
from part1.ridge_lasso import plot_ridge_trace, ridge_fit
from part2.advanced_methods import bayesian_linear_fit, bayesian_predict

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


def vif_values(X, feature_names=None):
    """
    Tinh VIF cho tung bien doc lap, bo qua cot intercept o index 0.

    X: 2D list da co cot intercept.
    feature_names: list ten cot, cung thu tu voi X. Neu None thi dung x1, x2, ...
    Returns: list of dict [{"feature": name, "VIF": value}, ...]
    """
    if not X or not X[0]:
        raise ValueError("X khong duoc rong va phai co it nhat 1 cot")

    n_cols = len(X[0])

    if n_cols < 2:
        return []

    results = []

    for j in range(1, n_cols):
        y_j = [[row[j]] for row in X]
        X_j = [[row[k] for k in range(n_cols) if k != j] for row in X]

        if feature_names is not None and j < len(feature_names):
            feature_name = feature_names[j]
        else:
            feature_name = f"x{j}"

        if len(X_j[0]) == 1:
            vif = 1.0
        else:
            beta_j, _ = ols_fit(X_j, y_j)
            y_hat_j = mat_mul(X_j, beta_j)

            p = len(X_j[0]) - 1
            metrics = model_metrics(y_j, y_hat_j, p)
            r2_j = metrics["R_squared"]

            denominator = 1 - r2_j

            if denominator <= 1e-12:
                vif = float("inf")
            else:
                vif = 1 / denominator

        results.append(
            {
                "feature": feature_name,
                "VIF": vif,
            }
        )

    return results


def ols_variable_selection(X, y, feature_names=None, p_threshold=0.05, vif_threshold=10.0):
    """
    Chon bien bang backward elimination dua tren VIF va p-value.

    X: 2D list da co cot intercept.
    y: 2D list cot target.
    feature_names: list ten cot, cung thu tu voi X.
    """
    if not X or not X[0]:
        raise ValueError("X khong duoc rong")
    if len(X) != len(y):
        raise ValueError("X va y phai co cung so dong")

    n_cols = len(X[0])
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_cols)]
    elif len(feature_names) != n_cols:
        raise ValueError("feature_names phai co cung so cot voi X")

    selected_indices = list(range(n_cols))
    removed_features = []
    removal_log = []

    while True:
        X_current = [[row[i] for i in selected_indices] for row in X]
        current_feature_names = [feature_names[i] for i in selected_indices]

        beta, sigma2 = ols_fit(X_current, y)

        if len(selected_indices) <= 1:
            break

        vifs = vif_values(X_current, current_feature_names)
        max_vif_item = max(vifs, key=lambda item: item["VIF"])

        if max_vif_item["VIF"] > vif_threshold:
            remove_name = max_vif_item["feature"]
            remove_position = current_feature_names.index(remove_name)
            remove_original_index = selected_indices[remove_position]

            selected_indices.remove(remove_original_index)
            removed_features.append(remove_name)
            removal_log.append({
                "step": len(removal_log) + 1,
                "feature": remove_name,
                "reason": "VIF",
                "value": float(max_vif_item["VIF"]),
            })
            continue

        inference = coef_inference(X_current, y, beta, sigma2)
        p_values = inference["p_values"]

        candidate_p_values = [
            {
                "feature": current_feature_names[i],
                "p_value": p_values[i],
                "position": i,
                "original_index": selected_indices[i],
            }
            for i in range(1, len(current_feature_names))
        ]
        max_p_item = max(candidate_p_values, key=lambda item: item["p_value"])

        if max_p_item["p_value"] > p_threshold:
            remove_name = max_p_item["feature"]
            remove_original_index = max_p_item["original_index"]

            selected_indices.remove(remove_original_index)
            removed_features.append(remove_name)
            removal_log.append({
                "step": len(removal_log) + 1,
                "feature": remove_name,
                "reason": "p-value",
                "value": float(max_p_item["p_value"]),
            })
            continue

        break

    X_final = [[row[i] for i in selected_indices] for row in X]
    beta_final, sigma2_final = ols_fit(X_final, y)

    return {
        "beta": beta_final,
        "sigma2": sigma2_final,
        "selected_features": [feature_names[i] for i in selected_indices],
        "selected_indices": selected_indices,
        "removed_features": removed_features,
        "removal_log": removal_log,
    }


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
    Train OLS co ban, OLS chon bien, Ridge(lambda tot nhat) va Bayesian
    Linear Regression, sau do danh gia tren test set.

    Returns:
        dict gom best_lambda, cv_results, metrics_table, selection_result,
        models, predictions, data
    """
    data = prepare_air_quality_data(data_path=data_path)

    best_lambda, cv_results = select_best_lambda(
        data["X_train"],
        data["y_train"],
        k=k,
        lambdas=lambdas,
    )

    beta_ols, sigma2_ols = ols_fit(data["X_train"], data["y_train"])
    selection_result = ols_variable_selection(
        data["X_train"],
        data["y_train"],
        data["feature_names"],
    )
    beta_ols_selected = selection_result["beta"]
    selected_indices = selection_result["selected_indices"]
    X_test_selected = [[row[i] for i in selected_indices] for row in data["X_test"]]

    beta_ridge = ridge_fit(data["X_train"], data["y_train"], best_lambda)
    posterior_bayes = bayesian_linear_fit(
        data["X_train"],
        data["y_train"],
        sigma2=sigma2_ols,
        prior_variance=100.0,
        intercept_prior_variance=1e12,
    )

    y_pred_ols = predict(data["X_test"], beta_ols)
    y_pred_ols_selected = predict(X_test_selected, beta_ols_selected)
    y_pred_ridge = predict(data["X_test"], beta_ridge)
    y_pred_bayes = bayesian_predict(data["X_test"], posterior_bayes)

    metrics_table = pd.DataFrame(
        [
            {"Model": "OLS", **compute_metrics(data["y_test"], y_pred_ols)},
            {
                "Model": "OLS (Variable Selection)",
                **compute_metrics(data["y_test"], y_pred_ols_selected),
            },
            {
                "Model": f"Ridge (lambda={best_lambda:.6g})",
                **compute_metrics(data["y_test"], y_pred_ridge),
            },
            {
                "Model": "Bayesian Linear Regression",
                **compute_metrics(data["y_test"], y_pred_bayes),
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
        "selection_result": selection_result,
        "models": {
            "ols": {"beta": beta_ols, "sigma2": sigma2_ols},
            "ols_selected": {
                "beta": beta_ols_selected,
                "sigma2": selection_result["sigma2"],
                "selected_indices": selected_indices,
            },
            "ridge": {"beta": beta_ridge, "lambda": best_lambda},
            "bayesian": posterior_bayes,
        },
        "predictions": {
            "ols": y_pred_ols,
            "ols_selected": y_pred_ols_selected,
            "ridge": y_pred_ridge,
            "bayesian": y_pred_bayes,
        },
        "data": data,
    }


def test_vif_no_multicollinearity():
    """VIF nho khi cac bien khong gan phu thuoc tuyen tinh."""
    X = [
        [1.0, -2.0, 4.0],
        [1.0, -1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 4.0],
        [1.0, 3.0, 9.0],
    ]

    vifs = vif_values(X, ["intercept", "x1", "x2"])

    assert all(item["VIF"] < 2.0 for item in vifs)
    print("TEST 1 PASSED: VIF thap khi khong co da cong tuyen manh")


def test_vif_high_multicollinearity():
    """VIF cao khi co 2 bien gan phu thuoc tuyen tinh."""
    X = []
    for i in range(1, 15):
        x1 = float(i)
        noise = 0.01 if i % 2 == 0 else -0.01
        x2 = x1 + noise
        X.append([1.0, x1, x2])

    vifs = vif_values(X, ["intercept", "x1", "x2"])

    assert any(item["VIF"] > 10.0 for item in vifs)
    print("TEST 2 PASSED: VIF cao khi bien gan trung tuyen tinh")


def test_variable_selection_removes_redundant():
    """OLS chon bien loai duoc bien da cong tuyen."""
    X = []
    y = []
    for i in range(1, 18):
        x1 = float(i)
        noise = 0.01 if i % 2 == 0 else -0.01
        x2 = x1 + noise
        X.append([1.0, x1, x2])
        y.append([1.0 + 2.0 * x1 + 0.05 * noise])

    result = ols_variable_selection(
        X,
        y,
        ["intercept", "x1", "x2"],
        p_threshold=1.0,
        vif_threshold=10.0,
    )

    assert len(result["removed_features"]) >= 1
    assert result["removal_log"][0]["reason"] == "VIF"
    print("TEST 3 PASSED: variable selection loai bien du thua theo VIF")


def test_variable_selection_keeps_significant():
    """OLS chon bien giu lai bien co y nghia."""
    X = []
    y = []
    for i in range(1, 18):
        x1 = float(i)
        x2 = 1.0 if i % 2 == 0 else -1.0
        noise = 0.05 if i % 3 == 0 else -0.03
        X.append([1.0, x1, x2])
        y.append([1.0 + 2.0 * x1 + noise])

    result = ols_variable_selection(
        X,
        y,
        ["intercept", "x1", "x2"],
        p_threshold=0.05,
        vif_threshold=10.0,
    )

    assert "x1" in result["selected_features"]
    print("TEST 4 PASSED: variable selection giu bien co y nghia")


def run_variable_selection_tests():
    test_vif_no_multicollinearity()
    test_vif_high_multicollinearity()
    test_variable_selection_removes_redundant()
    test_variable_selection_keeps_significant()


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
