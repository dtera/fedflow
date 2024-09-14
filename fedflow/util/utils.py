# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import glob
from enum import Enum
from os.path import dirname, basename, isfile, join
from typing import Any


class EnumBase(str, Enum):
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Enum):
            return self.value == other.value
        else:
            return self.value == other

    def __hash__(self):
        return hash(self.value)


def extend_enum(parent_enum, is_raw_enum=False):
    def wrapper(extended_enum):
        joined = {}
        for item in parent_enum:
            joined[item.name] = item.value
        for item in extended_enum:
            joined[item.name] = item.value
        return (Enum if is_raw_enum else EnumBase)(extended_enum.__name__, joined)

    return wrapper


def get_modules(f, exclude_fs=['__init__.py'], pre_modules=None):
    if exclude_fs is None:
        exclude_fs = []
    if '__init__.py' not in exclude_fs:
        exclude_fs = exclude_fs + ['__init__.py']
    if pre_modules is None:
        pre_modules = []
    modules = [
        basename(f)[:-3] for f in glob.glob(join(dirname(f), "*.py"))
        if isfile(f) and basename(f) not in exclude_fs
    ]
    # reorder the module
    for pre_module in pre_modules:
        modules.pop(modules.index(pre_module))
        modules.insert(0, pre_module)
    return modules
