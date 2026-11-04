"""Contract checks isolated from package exports during the model refactor."""

from pathlib import Path
import runpy

import numpy as np
import pytest
import xarray as xr


# Normal package imports still depend on the old vertical model names.
base = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/interpolate/models/base.py"))
Model = base["InterpolationModel"]
Fitted = base["FittedInterpolationModel"]


@pytest.fixture
def points():
    return xr.Dataset(
        {"elevation": ("target_id", [100.0, np.nan])},
        coords={"target_id": ["p", "q"], "x": ("target_id", [1.0, 2.0]),
                "y": ("target_id", [3.0, 4.0])},
    )


def test_training_aligns_by_id_without_sorting_or_filling():
    X = xr.Dataset({"elevation": ("id", [10.0, 20.0])}, coords={"id": ["a", "b"]})
    y = xr.DataArray([2.0, 1.0], dims="id", coords={"id": ["b", "a"]})
    aligned, values = Model._validate_training_data(X, y, ("elevation",))
    assert aligned.elevation.values.tolist() == [20.0, 10.0]
    xr.testing.assert_identical(values, y)
    with pytest.raises(ValueError, match="same source IDs"):
        Model._validate_training_data(X, y.assign_coords(id=["a", "c"]))


@pytest.mark.parametrize("ids", [["a", "a"], ["a", None]])
def test_training_rejects_ambiguous_ids(ids):
    X = xr.Dataset(coords={"id": ids})
    y = xr.DataArray([1.0, 2.0], dims="id", coords={"id": ids})
    with pytest.raises(ValueError, match="unique, nonmissing"):
        Model._validate_training_data(X, y)


def test_training_requires_explicit_ids_and_finite_samples():
    X = xr.Dataset({"elevation": ("id", [10.0])}, coords={"id": ["a"]})
    y = xr.DataArray([np.nan], dims="id", coords={"id": ["a"]})
    with pytest.raises(ValueError, match="non-finite"):
        Model._validate_training_data(X, y)
    with pytest.raises(ValueError, match="explicit"):
        Model._validate_training_data(X.drop_vars("id"), y)
    with pytest.raises(ValueError, match="only the 'id'"):
        Model._validate_training_data(X, y.rename(id="station"))


def test_target_layouts_and_predictors(points):
    assert Fitted._validate_targets(points, ("elevation",)) == ("target_id",)
    grid = xr.Dataset(
        {"elevation": (("x", "y"), [[1.0, np.nan], [3.0, 4.0]])},
        coords={"x": [1.0, 2.0], "y": [4.0, 3.0]},
    )
    assert Fitted._validate_targets(grid, ("elevation",)) == ("y", "x")
    with pytest.raises(ValueError, match="Missing required predictor"):
        Fitted._validate_targets(points.drop_vars("elevation"), ("elevation",))
    with pytest.raises(ValueError, match="Predictor 'elevation'"):
        Fitted._validate_targets(grid.assign(elevation=("x", [1.0, 2.0])), ("elevation",))
    with pytest.raises(ValueError, match="Targets must have dimensions"):
        Fitted._validate_targets(points.expand_dims(time=[0]))
    with pytest.raises(ValueError, match="coordinate 'x'"):
        Fitted._validate_targets(points.drop_vars("x"))


def point_distances(points):
    return xr.DataArray(
        [[1.0, np.nan, np.inf], [0.0, 2.0, 3.0]],
        dims=("target_id", "id"), coords={**points.coords, "id": ["b", "a", "extra"]},
    ).assign_coords(lam_value=0.5)


def test_distances_select_fitted_sources_and_preserve_unavailable_values(points):
    result = Fitted._validate_distances(point_distances(points), ["a", "b"], points)
    assert result.dims == ("id", "target_id")
    assert result.id.values.tolist() == ["a", "b"]
    assert np.isnan(result.values[0, 0])
    assert result.values[1, 1] == 0.0
    # Positive infinity is an allowed unavailable distance.
    Fitted._validate_distances(point_distances(points), ["extra"], points)


def test_distances_reject_missing_sources_and_wrong_destinations(points):
    distances = point_distances(points)
    with pytest.raises(ValueError, match="missing fitted source"):
        Fitted._validate_distances(distances, ["missing"], points)
    with pytest.raises(ValueError, match="Target coordinate 'target_id'"):
        Fitted._validate_distances(distances.isel(target_id=[1, 0]), ["a"], points)
    with pytest.raises(ValueError, match="Target coordinate 'x'"):
        Fitted._validate_distances(distances.assign_coords(x=("target_id", [9.0, 2.0])), ["a"], points)
    with pytest.raises(ValueError, match="select lambda/time"):
        Fitted._validate_distances(distances.expand_dims(time=[0]), ["a"], points)


@pytest.mark.parametrize("value", [-1.0, -np.inf])
def test_distances_reject_negative_values(points, value):
    distances = point_distances(points).copy()
    distances.values[0, 0] = value
    with pytest.raises(ValueError, match="nonnegative"):
        Fitted._validate_distances(distances, ["b"], points)


def test_prediction_wrapper_validates_results_and_keeps_metadata(points):
    class Example(Fitted):
        required_predictors = ('elevation',)

        def _predict(self, targets, distances=None):
            return targets.elevation.copy().assign_attrs(description="test")

    result = Example().predict(points)
    assert result.attrs["description"] == "test"
    assert np.isnan(result.values[1])
    with pytest.raises(ValueError, match="Missing required predictor"):
        Example().predict(points.drop_vars("elevation"))
    with pytest.raises(ValueError, match="Missing target coordinate 'x'"):
        Fitted._validate_predictions(result.drop_vars("x"), points)
    with pytest.raises(TypeError, match="DataArray"):
        Fitted._validate_predictions(None, points)


def test_grid_predictions_normalize_dimension_order():
    targets = xr.Dataset(coords={"y": [2.0, 1.0], "x": [3.0, 4.0, 5.0]})
    prediction = xr.DataArray(np.ones((3, 2)), dims=("x", "y"), coords=targets.coords)
    result = Fitted._validate_predictions(prediction, targets)
    assert result.dims == ("y", "x")
    with pytest.raises(ValueError, match="target dimensions"):
        Fitted._validate_predictions(prediction.expand_dims(id=["a"]), targets)


def test_grid_distances_require_the_same_grid():
    targets = xr.Dataset(coords={"y": [2.0, 1.0], "x": [3.0, 4.0, 5.0]})
    distances = xr.DataArray(
        np.ones((3, 1, 2)), dims=("x", "id", "y"),
        coords={**targets.coords, "id": ["a"]},
    )
    result = Fitted._validate_distances(distances, ["a"], targets)
    assert result.dims == ("id", "y", "x")
    with pytest.raises(ValueError, match="Target coordinate 'y'"):
        Fitted._validate_distances(distances.assign_coords(y=[1.0, 2.0]), ["a"], targets)
