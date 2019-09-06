import copy
from importlib import import_module

import yaml

from dataclasses import fields, InitVar, dataclass, field, replace, is_dataclass, _is_dataclass_instance
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Optional, Callable, Any, Union, Dict, Type, TypeVar, Sequence

HIDE = "hide"


class Backend(Enum):
    """
    list of available backends
    """

    PYTORCH = "pytorch"


class Representation(Enum):
    """
    list of available representations
    """

    # each available backend is also a valid representation
    PYTORCH = Backend.PYTORCH

    # in addition to backends the config data can be represented as strings
    STRING = "string"


Config = TypeVar("Config", bound="BaseConfig")


@dataclass
class BaseConfig:
    custom_module: Optional[Union[str, ModuleType]] = field(
        default=None, metadata={Representation(b): lambda name: import_module(name) if name else None for b in Backend}
    )
    repr: InitVar[Representation] = field(default=Representation.STRING, metadata={HIDE: True})
    impl: InitVar[Sequence[ModuleType]] = tuple()

    def __post_init__(self, repr: Representation, impl: Sequence[ModuleType]):

        for f in fields(self):
            v = getattr(self, f.name)

            if v is not None and (
                (not isinstance(v, str) and repr == Representation.STRING)
                or (isinstance(v, str) and repr != Representation.STRING)
            ):
                # mapping required
                map_to_fn = f.metadata.get(repr)
                if map_to_fn is None:
                    if isinstance(v, str):
                        # default mapping to a representation other than string
                        def map_to_fn(name: str):
                            for im in impl:
                                # check in dedicated submodule for `name`
                                ret = getattr(getattr(im, f.name, None), name, None)
                                if ret is not None:
                                    return ret

                                ret = getattr(im, name, None)
                                if ret is not None:
                                    return ret

                            return name

                    else:
                        # default mapping to string
                        map_to_fn = lambda obj: getattr(obj, "__name__", obj)

                new_value = map_to_fn(v)
                setattr(self, f.name, new_value)
                if f.name == "custom_module" and new_value:
                    # add custom module to sequence of implementation modules
                    impl = [new_value] + list(impl)

    def get_mapped(self, repr: Representation) -> Config:
        """
        Args:
            repr (Representation): Representation to map to. Always maps to `Representation.STRING` as intermediate.

        Returns:
            The Config for a specific Representation.
            Non-mappable objects are left unchanged, but all hidden fields should be mapped.
        """
        assert isinstance(repr, Representation)
        # map to string as intermediate representation
        intermediate = _get_mapped_inner(self, Representation.STRING, tuple())

        if repr == Representation.STRING:
            return intermediate
        elif repr == Representation.PYTORCH:
            from protozoo import impl_pytorch

            return _get_mapped_inner(intermediate, repr, [impl_pytorch])  # todo: fix typing of impl: ModuleType
        else:
            raise NotImplementedError(repr)

    def as_dict(self, dict_factory=dict, return_hidden=True):
        return _asdict_inner(self, dict_factory=dict_factory, return_hidden=return_hidden)

    def save(self, file_name: Path, repr: Representation = Representation.STRING, save_hidden=False, safe_dump=True):
        data = self.get_mapped(repr).as_dict(return_hidden=save_hidden)

        if safe_dump:
            dump = yaml.safe_dump
        else:
            dump = yaml.dump

        with file_name.open("w") as file:
            dump(data, file)

    @classmethod
    def load(
        cls: Type[Config], source: Union[Path, Dict[str, Any]], repr: Representation = Representation.STRING
    ) -> Config:
        if isinstance(source, Path):
            with source.open("r") as source_file:
                source = yaml.safe_load(source_file)

        return _load_inner(cls, source, repr)


T = TypeVar("T")


def _load_inner(type_: Type[T], source: dict, repr: Representation) -> T:
    if source is None or isinstance(source, (str, int, float)):
        return source

    try:
        is_BaseConfig = issubclass(type_, BaseConfig)
    except TypeError:
        is_BaseConfig = False

    if is_BaseConfig:
        source = {f.name: _load_inner(f.type, source[f.name], repr) for f in fields(type_) if f.name in source}

        return type_(repr=repr, **source)

    if isinstance(source, dict):
        # if `source` is a dict, and `type_` is not a `BaseConfig`, then `type_` is a type annotation for Dict/Mapping
        subtype = type_.__args__[1]  # type_ = typing.Dict[str, subtype]
        return {k: _load_inner(subtype, v, repr) for k, v in source.items()}

    if isinstance(source, list):
        # if `source` is a list, and `type_` is not a `BaseConfig`, then `type_` is a type annotation for List/Sequence
        subtype = type_.__args__[0]  # type_ = typing.List[subtype]
        return [_load_inner(subtype, v, repr) for v in source]

    raise NotImplementedError(type_, source)


def _get_mapped_inner(obj: T, repr: Representation, impl: Sequence[ModuleType]) -> T:
    if is_dataclass(obj):
        kwargs = {}
        for f in fields(obj):
            if f.init:
                v = getattr(obj, f.name)
                kwargs[f.name] = _get_mapped_inner(v, repr, impl)

        if isinstance(obj, BaseConfig):
            kwargs["repr"] = repr
            kwargs["impl"] = impl

        return replace(obj, **kwargs)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_get_mapped_inner(v, repr, impl) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_get_mapped_inner(k, repr, impl), _get_mapped_inner(v, repr, impl)) for k, v in obj.items())
    else:
        return obj


def _asdict_inner(obj: T, dict_factory: Callable, return_hidden: bool) -> T:
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
