from ..main import load_config


def test_cfg_dataset_single():
    cfg = load_config("camelsml/tests/test_dataset_single.txt")
    assert cfg["dataset"] == ["camels_gb"]


def test_cfg_dataset_multi():
    cfg = load_config("camelsml/tests/test_dataset_multi.txt")
    assert cfg["dataset"] == ["camels_gb", "camels_us"]


def test_default_dataset():
    cfg = load_config("camelsml/tests/test_dataset_none.txt")
    assert cfg["dataset"] == ["camels_gb"]


def test_load_timeseries_selection_gb():
    cfg = load_config("camelsml/tests/test_timeseries_selection_gb.txt")
    assert cfg["timeseries"] == [
        "precipitation",
        "temperature",
        "humidity",
        "shortwave_rad",
        "longwave_rad",
        "windspeed",
    ]


def test_load_timeseries_selection_us():
    cfg = load_config("camelsml/tests/test_timeseries_selection_us.txt")
    assert cfg["timeseries"] == [
        "prcp(mm/day)",
        "srad(W/m2)",
        "tmax(C)",
        "tmin(C)",
        "vp(Pa)",
    ]
