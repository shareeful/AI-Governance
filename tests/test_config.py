from __future__ import annotations

import pytest
import yaml

from agp.config import Config, require_disjoint, require_sum_to_one
from agp.errors import ConfigurationError, MissingResourceError


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_later_files_override_earlier_ones(tmp_path):
    base = _write(tmp_path, "base.yaml", {"a": {"b": 1, "c": 2}})
    override = _write(tmp_path, "over.yaml", {"a": {"b": 9}})
    config = Config.load(base, override)
    assert config.get("a.b") == 9
    assert config.get("a.c") == 2


def test_absent_required_key_is_reported_with_its_sources(tmp_path):
    config = Config.load(_write(tmp_path, "c.yaml", {"a": 1}))
    with pytest.raises(ConfigurationError) as error:
        config.get("missing.key")
    assert "missing.key" in str(error.value)


def test_null_value_is_treated_as_unset(tmp_path):
    config = Config.load(_write(tmp_path, "c.yaml", {"paths": {"root": None}}))
    with pytest.raises(ConfigurationError):
        config.get("paths.root")
    assert not config.has("paths.root")


def test_missing_path_is_refused_rather_than_substituted(tmp_path):
    config = Config.load(
        _write(tmp_path, "c.yaml", {"paths": {"root": str(tmp_path / "absent")}})
    )
    with pytest.raises(MissingResourceError):
        config.path("paths.root")


def test_probability_must_lie_strictly_inside_the_unit_interval(tmp_path):
    config = Config.load(_write(tmp_path, "c.yaml", {"a": {"x": 1.0, "y": 0.05}}))
    assert config.require_probability("a.y") == 0.05
    with pytest.raises(ConfigurationError):
        config.require_probability("a.x")


def test_empty_list_is_refused(tmp_path):
    config = Config.load(_write(tmp_path, "c.yaml", {"a": []}))
    with pytest.raises(ConfigurationError):
        config.list_value("a")


def test_weights_must_sum_to_one():
    require_sum_to_one({"a": 0.5, "b": 0.5}, 1e-9, "weights")
    with pytest.raises(ConfigurationError):
        require_sum_to_one({"a": 0.5, "b": 0.4}, 1e-9, "weights")


def test_splits_must_be_disjoint():
    require_disjoint({"train": [1, 2], "test": [3]}, "splits")
    with pytest.raises(ConfigurationError):
        require_disjoint({"train": [1, 2], "test": [2]}, "splits")
