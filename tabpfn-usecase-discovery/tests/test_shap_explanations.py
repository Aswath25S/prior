import numpy as np
import pytest

from src.shap_explanations import interventional_shap


def test_linear_shap_matches_known_analytic_contributions():
    background = np.array([[0, 1, 5], [2, 3, 7]], dtype=float)
    row = np.array([3, 0, 2], dtype=float)
    coefficients = np.array([2, -4, 0], dtype=float)
    values, base, predicted, method = interventional_shap(lambda X: 10 + X @ coefficients, background, row)
    assert np.allclose(values, coefficients * (row - background.mean(axis=0)))
    assert predicted == pytest.approx(10 + row @ coefficients)
    assert base + values.sum() == pytest.approx(predicted)
    assert method.startswith("Exact")


def test_shap_shares_nonlinear_interaction_instead_of_relabeling_sensitivity():
    values, base, predicted, _ = interventional_shap(lambda X: X[:, 0] * X[:, 1], np.zeros((4, 2)), np.array([2, 3]))
    # Single-field replacement would claim +6 for each field; SHAP shares the +6.
    assert np.allclose(values, [3, 3])
    assert base == 0 and predicted == 6


def test_estimated_shap_preserves_additivity_and_dummy_fields():
    background = np.zeros((8, 9))
    row = np.ones(9)
    values, base, predicted, method = interventional_shap(lambda X: 3 + 2 * X[:, 0] * X[:, 1], background, row)
    assert np.allclose(values[:2], [1, 1])
    assert np.allclose(values[2:], 0)
    assert base + values.sum() == pytest.approx(predicted)
    assert method.startswith("Estimated")


def test_shap_batches_requests_and_is_reproducible():
    rng = np.random.default_rng(8)
    background, row = rng.normal(size=(8, 12)), rng.normal(size=12)
    calls = []

    def predict(X):
        calls.append(len(X))
        return np.sin(X[:, 0]) + X[:, 3] * X[:, 5]

    first = interventional_shap(predict, background, row, batch_size=64)
    second = interventional_shap(predict, background, row, batch_size=64)
    assert max(calls) <= 64
    assert np.allclose(first[0], second[0])
    assert first[1] + first[0].sum() == pytest.approx(first[2])


def test_invalid_model_outputs_fail_instead_of_displaying_fake_shap():
    with pytest.raises(ValueError, match="finite model output"):
        interventional_shap(lambda X: np.full(len(X), np.nan), np.zeros((8, 2)), np.ones(2))
