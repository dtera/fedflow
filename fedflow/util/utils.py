# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import glob
from os.path import dirname, basename, isfile, join


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
