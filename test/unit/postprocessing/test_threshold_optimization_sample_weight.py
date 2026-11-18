# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Tests for the sample_weight support of ThresholdOptimizer."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.postprocessing._constants import (
    LABEL_KEY,
    SAMPLE_WEIGHT_KEY,
    SAMPLE_WEIGHT_NOT_SUPPORTED_WARNING,
    SCORE_KEY,
)
from fairlearn.postprocessing._threshold_optimizer import (
    INVALID_SAMPLE_WEIGHT_ERROR_MESSAGE,
    SAMPLE_WEIGHT_LENGTH_ERROR_MESSAGE,
    ZERO_WEIGHT_GROUP_ERROR_MESSAGE,
    _validate_sample_weight,
)
from fairlearn.postprocessing._tradeoff_curve_utilities import _calculate_tradeoff_points


class PassThroughPredictor(BaseEstimator):
    def __init__(self, transform=None):
        self.transform = transform

    def fit(self, X, y=None, sample_weight=None, **kwargs):
        self.transform_ = self.transform
        return self

    def predict(self, X):
        scores = X[0]
        if self.transform_ is not None:
            scores = self.transform_(scores)
        return scores


class NoSampleWeightPredictor(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[0]


class WeightRecordingPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = sample_weight
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]


def _assert_same_solution(left, right):
    left_dict = left.interpolated_thresholder_.interpolation_dict
    right_dict = right.interpolated_thresholder_.interpolation_dict
    assert set(left_dict) == set(right_dict)
    for key in left_dict:
        assert left_dict[key]["p0"] == pytest.approx(right_dict[key]["p0"])
        assert left_dict[key]["p1"] == pytest.approx(right_dict[key]["p1"])
        for operation in ["operation0", "operation1"]:
            assert (
                left_dict[key][operation].operator == right_dict[key][operation].operator
            )
            assert left_dict[key][operation].threshold == pytest.approx(
                right_dict[key][operation].threshold
            )
        assert ("p_ignore" in left_dict[key]) == ("p_ignore" in right_dict[key])
        if "p_ignore" in left_dict[key]:
            assert left_dict[key]["p_ignore"] == pytest.approx(right_dict[key]["p_ignore"])
            assert left_dict[key]["prediction_constant"] == pytest.approx(
                right_dict[key]["prediction_constant"]
            )


@pytest.fixture
def data():
    X = pd.DataFrame(np.arange(12).reshape(-1, 1))
    y = pd.Series([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    sensitive_features = pd.Series([0] * 6 + [1] * 6)
    weights = np.array([1, 3, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2], dtype=float)
    return X, y, sensitive_features, weights


@pytest.mark.parametrize("constraints", ["demographic_parity", "equalized_odds"])
def test_weighted_equals_replicated_data(data, constraints):
    X, y, sensitive_features, weights = data
    repeated_indices = np.repeat(np.arange(len(y)), weights.astype(int))

    weighted = ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints=constraints, grid_size=20
    ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)

    replicated = ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints=constraints, grid_size=20
    ).fit(
        X.iloc[repeated_indices].reset_index(drop=True),
        y.iloc[repeated_indices].reset_index(drop=True),
        sensitive_features=sensitive_features.iloc[repeated_indices].reset_index(drop=True),
    )

    _assert_same_solution(weighted, replicated)
    for value in weighted._tradeoff_curve:
        np.testing.assert_allclose(
            weighted._tradeoff_curve[value]["y"], replicated._tradeoff_curve[value]["y"]
        )


@pytest.mark.parametrize("constraints", ["demographic_parity", "equalized_odds"])
def test_zero_weight_samples_do_not_change_solution(data, constraints):
    X, y, sensitive_features, weights = data

    extra_X = pd.concat([X, X.iloc[:4]], ignore_index=True)
    extra_y = pd.concat([y, y.iloc[:4]], ignore_index=True)
    extra_sensitive_features = pd.concat(
        [sensitive_features, sensitive_features.iloc[:4]], ignore_index=True
    )
    extra_weights = np.concatenate([weights, np.zeros(4)])

    weighted = ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints=constraints, grid_size=20
    ).fit(
        extra_X,
        extra_y,
        sensitive_features=extra_sensitive_features,
        sample_weight=extra_weights,
    )

    reference = ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints=constraints, grid_size=20
    ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)

    _assert_same_solution(weighted, reference)


@pytest.mark.parametrize("constraints", ["demographic_parity", "equalized_odds"])
def test_zero_weight_samples_do_not_change_predictions(data, constraints):
    X, y, sensitive_features, weights = data
    extra_X = pd.concat([X, X.iloc[:4]], ignore_index=True)
    extra_y = pd.concat([y, y.iloc[:4]], ignore_index=True)
    extra_sensitive_features = pd.concat(
        [sensitive_features, sensitive_features.iloc[:4]], ignore_index=True
    )

    reference = ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints=constraints, grid_size=20
    ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)
    weighted = ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints=constraints, grid_size=20
    ).fit(
        extra_X,
        extra_y,
        sensitive_features=extra_sensitive_features,
        sample_weight=np.concatenate([weights, np.zeros(4)]),
    )

    reference_pmf = reference._pmf_predict(X, sensitive_features=sensitive_features)
    weighted_pmf = weighted._pmf_predict(X, sensitive_features=sensitive_features)
    np.testing.assert_allclose(reference_pmf, weighted_pmf)


@pytest.mark.parametrize("constraints", ["demographic_parity", "equalized_odds"])
def test_zero_weight_group_rejected(data, constraints):
    X, y, sensitive_features, weights = data
    zero_group_weights = weights.copy()
    zero_group_weights[sensitive_features == 1] = 0.0

    with pytest.raises(ValueError, match="total sample weight"):
        ThresholdOptimizer(
            estimator=PassThroughPredictor(), constraints=constraints
        ).fit(X, y, sensitive_features=sensitive_features, sample_weight=zero_group_weights)


def test_all_zero_weights_rejected(data):
    X, y, sensitive_features, weights = data
    with pytest.raises(ValueError, match="total sample weight"):
        ThresholdOptimizer(estimator=PassThroughPredictor()).fit(
            X, y, sensitive_features=sensitive_features, sample_weight=np.zeros(len(y))
        )


@pytest.mark.parametrize(
    "invalid_weight",
    [
        pytest.param(-1.0, id="negative"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="inf"),
        pytest.param(-np.inf, id="negative-inf"),
    ],
)
def test_invalid_sample_weight_values_rejected(data, invalid_weight):
    X, y, sensitive_features, weights = data
    invalid = weights.copy()
    invalid[3] = invalid_weight

    with pytest.raises(ValueError, match=INVALID_SAMPLE_WEIGHT_ERROR_MESSAGE.split("{")[0]):
        ThresholdOptimizer(estimator=PassThroughPredictor()).fit(
            X, y, sensitive_features=sensitive_features, sample_weight=invalid
        )


def test_invalid_sample_weight_shape_rejected(data):
    X, y, sensitive_features, weights = data
    invalid = np.vstack([weights, weights])

    with pytest.raises(ValueError, match=INVALID_SAMPLE_WEIGHT_ERROR_MESSAGE.split("{")[0]):
        ThresholdOptimizer(estimator=PassThroughPredictor()).fit(
            X, y, sensitive_features=sensitive_features, sample_weight=invalid
        )


def test_invalid_sample_weight_length_rejected(data):
    X, y, sensitive_features, weights = data

    with pytest.raises(ValueError, match=SAMPLE_WEIGHT_LENGTH_ERROR_MESSAGE.split("\\")[0]):
        ThresholdOptimizer(estimator=PassThroughPredictor()).fit(
            X, y, sensitive_features=sensitive_features, sample_weight=weights[:-1]
        )


def test_sample_weight_forwarded_to_base_estimator(data):
    X, y, sensitive_features, weights = data
    optimizer = ThresholdOptimizer(
        estimator=WeightRecordingPredictor(), constraints="demographic_parity"
    ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)

    np.testing.assert_allclose(optimizer.estimator_.sample_weight_, weights)


def test_sample_weight_not_supported_by_base_estimator_warns(data):
    X, y, sensitive_features, weights = data

    # NoSampleWeightPredictor.fit does not accept sample_weight, so has_fit_parameter is False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ThresholdOptimizer(estimator=NoSampleWeightPredictor()).fit(
            X, y, sensitive_features=sensitive_features, sample_weight=weights
        )

    messages = [str(warning.message) for warning in caught]
    assert any(SAMPLE_WEIGHT_NOT_SUPPORTED_WARNING.format("NoSampleWeightPredictor") in m
               for m in messages)


def test_sample_weight_with_prefit_estimator(data):
    X, y, sensitive_features, weights = data
    estimator = PassThroughPredictor().fit(X, y)

    optimizer = ThresholdOptimizer(estimator=estimator, prefit=True).fit(
        X, y, sensitive_features=sensitive_features, sample_weight=weights
    )
    predictions = optimizer.predict(X, sensitive_features=sensitive_features)
    assert set(predictions).issubset({0, 1})


@pytest.mark.parametrize("constraints", ["demographic_parity", "equalized_odds"])
def test_constraint_holds_on_weighted_predictions(data, constraints):
    X, y, sensitive_features, weights = data
    optimizer = ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints=constraints, grid_size=50
    ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)

    positive_probs = optimizer._pmf_predict(
        X, sensitive_features=sensitive_features
    )[:, 1]

    if constraints == "demographic_parity":
        weighted_selection_rates = {}
        for value in sensitive_features.unique():
            mask = (sensitive_features == value).to_numpy()
            weighted_selection_rates[value] = np.average(
                positive_probs[mask], weights=weights[mask]
            )
        np.testing.assert_allclose(list(weighted_selection_rates.values()),
                                   [np.mean(list(weighted_selection_rates.values()))] * 2)
    else:
        weighted_positive_rates = {}
        weighted_negative_rates = {}
        for value in sensitive_features.unique():
            group_mask = (sensitive_features == value).to_numpy()
            pos_mask = group_mask & (y == 1).to_numpy()
            neg_mask = group_mask & (y == 0).to_numpy()
            weighted_positive_rates[value] = np.average(
                positive_probs[pos_mask], weights=weights[pos_mask]
            )
            weighted_negative_rates[value] = np.average(
                positive_probs[neg_mask], weights=weights[neg_mask]
            )
        np.testing.assert_allclose(list(weighted_positive_rates.values()),
                                   [np.mean(list(weighted_positive_rates.values()))] * 2)
        np.testing.assert_allclose(list(weighted_negative_rates.values()),
                                   [np.mean(list(weighted_negative_rates.values()))] * 2)


def test_group_proportions_use_weights(data):
    X, y, sensitive_features, weights = data
    optimizer = ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints="demographic_parity", grid_size=10
    ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)

    unweighted_optimizer = ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints="demographic_parity", grid_size=10
    ).fit(X, y, sensitive_features=sensitive_features)

    expected_proportions = {
        value: weights[(sensitive_features == value).to_numpy()].sum() / weights.sum()
        for value in sensitive_features.unique()
    }
    # The overall curve must be the weighted average of the group curves.
    np.testing.assert_allclose(
        optimizer._overall_tradeoff_curve["y"],
        sum(
            expected_proportions[value] * optimizer._tradeoff_curve[value]["y"]
            for value in expected_proportions
        ),
    )
    # The weighting is not trivial: unweighted proportions differ here.
    assert not np.allclose(
        optimizer._overall_tradeoff_curve["y"],
        unweighted_optimizer._overall_tradeoff_curve["y"],
    )


def test_relaxed_constraints_use_weights(data):
    X, y, sensitive_features, weights = data
    optimizer = ThresholdOptimizer(
        estimator=PassThroughPredictor(),
        constraints="demographic_parity",
        grid_size=20,
        tol=0.1,
    ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)

    expected_proportions = {
        value: weights[(sensitive_features == value).to_numpy()].sum() / weights.sum()
        for value in sensitive_features.unique()
    }
    # The selected objective equals the weighted objective of the group curves.
    selected_indices = [
        optimizer._x_grid.searchsorted(optimizer._x_best_per_group[value])
        for value in optimizer._x_best_per_group
    ]
    weighted_objective = sum(
        expected_proportions[value]
        * optimizer._tradeoff_curve[value]["y"].iloc[selected_indices[i]]
        for i, value in enumerate(optimizer._x_best_per_group)
    )
    assert optimizer._y_best == pytest.approx(weighted_objective)


def test_weighted_equivalence_for_all_simple_constraints(data):
    X, y, sensitive_features, weights = data
    for constraints in [
        "selection_rate_parity",
        "false_positive_rate_parity",
        "false_negative_rate_parity",
        "true_positive_rate_parity",
        "true_negative_rate_parity",
    ]:
        repeated_indices = np.repeat(np.arange(len(y)), weights.astype(int))
        weighted = ThresholdOptimizer(
            estimator=PassThroughPredictor(), constraints=constraints, grid_size=20
        ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)
        replicated = ThresholdOptimizer(
            estimator=PassThroughPredictor(), constraints=constraints, grid_size=20
        ).fit(
            X.iloc[repeated_indices].reset_index(drop=True),
            y.iloc[repeated_indices].reset_index(drop=True),
            sensitive_features=sensitive_features.iloc[repeated_indices].reset_index(drop=True),
        )
        _assert_same_solution(weighted, replicated)


def test_weighted_objectives(data):
    X, y, sensitive_features, weights = data
    for objective in [
        "accuracy_score",
        "balanced_accuracy_score",
        "selection_rate",
        "true_positive_rate",
        "true_negative_rate",
    ]:
        repeated_indices = np.repeat(np.arange(len(y)), weights.astype(int))
        weighted = ThresholdOptimizer(
            estimator=PassThroughPredictor(),
            constraints="demographic_parity",
            objective=objective,
            grid_size=20,
        ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)
        replicated = ThresholdOptimizer(
            estimator=PassThroughPredictor(),
            constraints="demographic_parity",
            objective=objective,
            grid_size=20,
        ).fit(
            X.iloc[repeated_indices].reset_index(drop=True),
            y.iloc[repeated_indices].reset_index(drop=True),
            sensitive_features=sensitive_features.iloc[repeated_indices].reset_index(drop=True),
        )
        _assert_same_solution(weighted, replicated)


def test_calculate_tradeoff_points_with_weights():
    data = pd.DataFrame(
        {
            SCORE_KEY: [3.0, 2.0, 1.0, 0.0],
            LABEL_KEY: [1, 0, 1, 0],
            SAMPLE_WEIGHT_KEY: [1.0, 1.0, 3.0, 1.0],
        }
    )
    weighted_points = _calculate_tradeoff_points(data, "group", flip=False)
    # Replicate the samples by their (integer) weights.
    replicated = data.loc[data.index.repeat(data[SAMPLE_WEIGHT_KEY].astype(int))].drop(
        columns=[SAMPLE_WEIGHT_KEY]
    )
    replicated_points = _calculate_tradeoff_points(replicated, "group", flip=False)

    np.testing.assert_allclose(weighted_points["x"], replicated_points["x"])
    np.testing.assert_allclose(weighted_points["y"], replicated_points["y"])
    assert [
        (op.operator, op.threshold) for op in weighted_points["operation"]
    ] == [(op.operator, op.threshold) for op in replicated_points["operation"]]


def test_calculate_tradeoff_points_ignores_zero_weight_samples():
    data = pd.DataFrame(
        {
            SCORE_KEY: [3.0, 2.0, 1.0, 0.0],
            LABEL_KEY: [1, 0, 1, 0],
            SAMPLE_WEIGHT_KEY: [1.0, 1.0, 1.0, 0.0],
        }
    )
    without_zero_weight = data[data[SAMPLE_WEIGHT_KEY] > 0].drop(columns=[SAMPLE_WEIGHT_KEY])

    weighted_points = _calculate_tradeoff_points(data, "group", flip=True)
    reference_points = _calculate_tradeoff_points(without_zero_weight, "group", flip=True)

    np.testing.assert_allclose(weighted_points["x"], reference_points["x"])
    np.testing.assert_allclose(weighted_points["y"], reference_points["y"])
    assert [
        (op.operator, op.threshold) for op in weighted_points["operation"]
    ] == [(op.operator, op.threshold) for op in reference_points["operation"]]


def test_calculate_tradeoff_points_degenerate_weighted_labels():
    data = pd.DataFrame(
        {
            SCORE_KEY: [1.0, 0.0],
            LABEL_KEY: [1, 0],
            SAMPLE_WEIGHT_KEY: [1.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="Degenerate labels"):
        _calculate_tradeoff_points(data, "group")


def test_validate_sample_weight_accepts_common_formats():
    expected = np.array([1.0, 2.0, 3.0])
    for weights in [expected, list(expected), pd.Series(expected), expected.reshape(-1, 1)]:
        validated = _validate_sample_weight(weights, expected_length=3)
        np.testing.assert_array_equal(validated, expected)
    assert _validate_sample_weight(None, expected_length=3) is None


def test_sample_weight_does_not_modify_callers_array(data):
    X, y, sensitive_features, weights = data
    original = weights.copy()
    ThresholdOptimizer(
        estimator=PassThroughPredictor(), constraints="demographic_parity"
    ).fit(X, y, sensitive_features=sensitive_features, sample_weight=weights)
    np.testing.assert_array_equal(weights, original)


def test_non_numeric_sample_weight_rejected(data):
    X, y, sensitive_features, weights = data
    with pytest.raises(ValueError, match=INVALID_SAMPLE_WEIGHT_ERROR_MESSAGE.split("{")[0]):
        ThresholdOptimizer(estimator=PassThroughPredictor()).fit(
            X, y, sensitive_features=sensitive_features, sample_weight=["a"] * len(y)
        )
