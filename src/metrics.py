import numpy as np

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))

def tolerance_accuracy_months(y_true, y_pred, tol_months: int) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    ok = np.abs(y_true - y_pred) <= tol_months
    return float(np.mean(ok))
