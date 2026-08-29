from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robustness_benchmark.methods.roar_lime import (
    hard_label_probabilities,
    input_space_parameters,
    logistic_lime_parameters,
    optimize_roar,
    worst_case_parameter_shift,
)


def test_extracts_full_logistic_vector_from_lime_compatibility_shape():
    explanation = SimpleNamespace(
        local_exp={1: [(0, np.array([[1.5, -2.0, 0.25]]))]},
        intercept={1: np.array([-0.75])},
    )

    coefficients, intercept = logistic_lime_parameters(explanation, 3)

    np.testing.assert_allclose(coefficients, [1.5, -2.0, 0.25])
    assert intercept == -0.75


def test_hard_label_probabilities_preserve_binary_lime_shape():
    probabilities = np.array([[0.8, 0.2], [0.5, 0.5], [0.01, 0.99]])

    labels = hard_label_probabilities(probabilities)

    np.testing.assert_array_equal(labels, [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])


def test_converts_lime_parameters_back_to_input_coordinates():
    coefficients, intercept = input_space_parameters(
        np.array([2.0, -3.0]),
        0.5,
        np.array([2.0, 0.5]),
        np.array([4.0, -1.0]),
    )

    np.testing.assert_allclose(coefficients, [1.0, -6.0])
    assert intercept == pytest.approx(-9.5)

    raw = np.array([5.0, 2.0])
    scaled = (raw - np.array([4.0, -1.0])) / np.array([2.0, 0.5])
    assert coefficients @ raw + intercept == pytest.approx(
        np.array([2.0, -3.0]) @ scaled + 0.5
    )


def test_worst_case_shift_minimizes_target_one_logit():
    values = torch.tensor([2.0, -3.0, 0.0])

    weight_shift, intercept_shift = worst_case_parameter_shift(
        values, target=1, delta=0.1
    )

    torch.testing.assert_close(weight_shift, torch.tensor([-0.1, 0.1, -0.0]))
    assert float(intercept_shift) == pytest.approx(-0.1)


def test_roar_optimizer_crosses_worst_case_linear_boundary():
    counterfactual, iterations, _, _, _ = optimize_roar(
        np.array([-1.0]),
        np.array([2.0]),
        0.0,
        target=1,
        delta=0.1,
        lambda_param=0.01,
        learning_rate=0.02,
        loss_threshold=1e-6,
        maximum_iterations=5_000,
    )

    worst_logit = (2.0 - 0.1) * counterfactual[0] - 0.1
    assert iterations <= 5_000
    assert worst_logit > 0.0
