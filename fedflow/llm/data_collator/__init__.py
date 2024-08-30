# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.

from fedflow.register import (
    register_module
)
from fedflow.util import get_modules

# to register the core-config before set up the default config
data_collator = get_modules(__file__)

register_module("data_collator", data_collator)
