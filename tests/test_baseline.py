import numpy as np

from expression_copilot._torch_regression import TorchMLPRegressor
from expression_copilot.baseline import Baseline


def test_baselines():
    a = np.random.rand(500, 50).astype(np.float32)
    b = np.random.rand(500, 40).astype(np.float32)
    for method in ["linear", "ridge", "mlp", "ensemble"]:
        model = Baseline(a, b, method=method)
        model.fit()
        y_pred = model.transform()
        metrics = model.cal_metrics(model.y_test, y_pred)
        print(f"Method: {method}, metrics: \n{metrics.mean()}")


def test_torch_baseline():
    a = np.random.rand(500, 50).astype(np.float32)
    b = np.random.rand(500, 40).astype(np.float32)
    model = TorchMLPRegressor()
    model.fit(a, b)
    y_pred = model.predict(a)
    assert y_pred.shape == b.shape
    print("Torch MLP Regressor test passed.")


if __name__ == "__main__":
    # test_baselines()
    test_torch_baseline()
    print("Baseline model test passed.")
