import copy
import yaml

from dataclasses import fields, InitVar, dataclass, field, replace, is_dataclass, _is_dataclass_instance
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any

HIDE = "hide"


class MapTo(Enum):
    STRING = "map_to_string"
    PYTORCH = "map_to_pytorch"


@dataclass
class BaseConfig:
    map_to: InitVar[MapTo] = field(default=MapTo.STRING, metadata={HIDE: True})

    def __post_init__(self, map_to):
        for f in fields(self):
            v = getattr(self, f.name)
            map_to_fn = f.metadata.get(map_to, None)
            if map_to_fn is not None:
                if (map_to == MapTo.STRING and not isinstance(v, str)) or (
                    map_to != MapTo.STRING and isinstance(v, str)
                ):
                    setattr(self, f.name, map_to_fn(v))

    def get_mapped(self, map_to: MapTo):
        return _get_mapped_inner(self, map_to)

    def as_dict(self, dict_factory=dict, return_hidden=True):
        return _asdict_inner(self, dict_factory=dict_factory, return_hidden=return_hidden)

    def export(self, file_name: Path, map_to: Optional[MapTo] = None, export_hidden=False, safe_dump=True):
        if map_to is not None:
            exp = self.get_mapped(map_to)
        else:
            exp = self

        exp = exp.as_dict(return_hidden=export_hidden)

        if safe_dump:
            dump = yaml.safe_dump
        else:
            dump = yaml.dump

        with file_name.open("w") as file:
            dump(exp, file)


def _get_mapped_inner(obj: Any, map_to: MapTo):
    if is_dataclass(obj):
        kwargs = {}
        for f in fields(obj):
            if f.init:
                v = getattr(obj, f.name)
                kwargs[f.name] = _get_mapped_inner(v, map_to)

        if isinstance(obj, BaseConfig):
            kwargs["map_to"] = map_to

        return replace(obj, **kwargs)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_get_mapped_inner(v, map_to) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_get_mapped_inner(k, map_to), _get_mapped_inner(v, map_to)) for k, v in obj.items())
    else:
        return obj


def _asdict_inner(obj: Any, dict_factory: Callable, return_hidden: bool):
    """
    Exactly like dataclasses.asdict(), except that it allows to filter fields by specified metadata
    """
    if _is_dataclass_instance(obj):
        result = []
        for f in fields(obj):
            # >>> Added filter by HIDE flag in metadata
            if not return_hidden and f.metadata.get(HIDE, False):
                continue
            # <<<
            value = _asdict_inner(getattr(obj, f.name), dict_factory, return_hidden)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The the main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_inner(v, dict_factory, return_hidden) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_asdict_inner(k, dict_factory, return_hidden), _asdict_inner(v, dict_factory, return_hidden))
            for k, v in obj.items()
        )
    else:
        return copy.deepcopy(obj)


class MiniBatch:
    """
    Base class for different mini batch formats. Mainly used for type checking
    """

    pass
