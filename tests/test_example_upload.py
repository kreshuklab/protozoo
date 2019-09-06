from pathlib import Path
from tempfile import TemporaryDirectory

from exampleupload.modelzoo import get_entry
from protozoo.config_base import BaseConfig, Representation
from protozoo.entry import ModelZooEntry


def test_entry_as_dict():
    entry = get_entry()
    entry_dict = entry.as_dict()
    assert isinstance(entry_dict, dict)
    for k, v in entry_dict.items():
        assert not isinstance(v, BaseConfig)


def test_map_entry():
    entry = get_entry()
    entry_pytorch = entry.get_mapped(Representation.PYTORCH)
    entry_pytorch.get_mapped(Representation.STRING)


def test_save_load():
    entry = get_entry()
    with TemporaryDirectory() as tmp_dir:
        file_name = Path(tmp_dir) / "test.yaml"
        entry.save(file_name)
        loaded = ModelZooEntry.load(file_name)

    assert isinstance(loaded, ModelZooEntry)
