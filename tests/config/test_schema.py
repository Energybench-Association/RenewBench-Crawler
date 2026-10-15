# tests/config/test_schema.py
from pathlib import Path

import pytest
from pydantic import ValidationError

from rbc.config.schema import SCHEMA_REGISTRY, Paths


@pytest.mark.parametrize("source", list(SCHEMA_REGISTRY.keys()))
def test_config_validates(source: str, source_configs: dict) -> None:
    """Happy path for data source schemas.

    Check the config output is valid given a set of expected inputs.

    Args:
        source (str): Name of the data source.
        source_configs (dict): Dictionary of all source configurations.
    """
    schema = SCHEMA_REGISTRY[source]
    cfg_dict = source_configs[source]

    received_cfg_obj = schema.model_validate({"source": source, **cfg_dict})
    received_cfg_dict = received_cfg_obj.model_dump(mode="json")

    assert received_cfg_obj.source == source
    for k, exp_v in cfg_dict.items():
        if isinstance(exp_v, dict):
            for sub_k, sub_v in exp_v.items():
                received_v = received_cfg_dict[k][sub_k]
                if isinstance(sub_v, str) and ("/" in sub_v or "\\" in sub_v):
                    assert Path(received_v) == Path(sub_v)
                else:
                    assert received_v == sub_v
        else:
            assert received_cfg_dict[k] == exp_v


@pytest.mark.parametrize("source", list(SCHEMA_REGISTRY.keys()))
@pytest.mark.parametrize("bad", ["", "   ", "YOUR-SECRET", "COMMIT"])
def test_config_with_access_rejects_placeholders(
    source: str, source_configs: dict, bad: str
) -> None:
    """Failure path for data source schemas with "access" fields.

    Check that schemas with access requirements reject placeholders or empty
    values.

    Args:
        source (str): Name of the data source.
        source_configs (dict): Dictionary of all source configurations.
        bad (str): Possible bad access string (placeholder or empty).
    """
    schema = SCHEMA_REGISTRY[source]
    cfg_dict = source_configs[source]

    if not cfg_dict.get("access"):
        pytest.skip(f"Source '{source}' has no access block. Skipped.")

    bad_cfg_dict = {
        "source": source,
        **cfg_dict,
        "access": {k: bad for k in cfg_dict["access"].keys()},
    }

    with pytest.raises(ValidationError):
        schema.model_validate(bad_cfg_dict)


def test_configured_proc_dir_is_kept(tmp_path: Path) -> None:
    """Happy path: a configured dst_dir_proc is used as is, nothing is derived.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
    """
    paths = Paths(
        dst_dir_raw=Path(tmp_path, "raw", "ons"), dst_dir_proc=Path(tmp_path, "mine")
    )
    assert paths.dst_dir_proc == Path(tmp_path, "mine")


@pytest.mark.parametrize(
    "raw, expected",
    [
        (("data", "raw", "ons"), ("data", "processed", "ons")),  # "raw" folder swapped
        (("outputs2", "ons"), ("outputs2", "processed", "ons")),  # else beside raw dir
    ],
)
def test_proc_dir_defaults_beside_raw_dir(
    tmp_path: Path, raw: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    """Happy path: without a configured dst_dir_proc, it's derived from dst_dir_raw.

    Without a "raw" folder to swap, it goes beside dst_dir_raw: the downloader created
    dst_dir_raw there, so its parent is writable. Two levels up wasn't (e.g. a cluster's
    shared workspace root), and as every config creates its paths on loading, that broke
    even the download scripts.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.
        raw (tuple[str, ...]): Path parts of dst_dir_raw below `tmp_path`.
        expected (tuple[str, ...]): Path parts of the derived dst_dir_proc.
    """
    assert Paths(dst_dir_raw=Path(tmp_path, *raw)).dst_dir_proc == Path(
        tmp_path, *expected
    )


def test_config_leaves_proc_dir_to_processing(
    source_configs: dict, tmp_path: Path
) -> None:
    """Happy path: loading a config creates dst_dir_raw, but not dst_dir_proc.

    Every script loads a config, downloads too, while only processing writes into
    dst_dir_proc (and creates it then). Creating it on loading left empty folders behind
    for every operator that was only downloaded.

    Args:
        source_configs (dict): Dictionary of all source configurations.
        tmp_path (Path): Pytest-provided temporary directory (holds dst_dir_raw).
    """
    cfg = SCHEMA_REGISTRY["ons"].model_validate(
        {"source": "ons", **source_configs["ons"]}
    )

    assert Path(tmp_path, "ons").is_dir()
    assert cfg.paths.dst_dir_proc == Path(tmp_path, "processed", "ons")
    assert not cfg.paths.dst_dir_proc.exists()
