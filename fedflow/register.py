# coding: utf-8
# Copyright (c) dterazhao. All rights reserved.
import inspect
import logging
import queue

logger = logging.getLogger(__name__)

datas = {}

data_collators = {}

models = {}

trainers = {}

args = {}

commons = {}

send_queue = queue.SimpleQueue()
recv_queue = queue.SimpleQueue()


def register(key, module, module_dict):
    if key in module_dict:
        logger.warning('Key {} is already pre-defined, overwritten.'.format(key))
    module_dict[key] = module


for k, v in [(k, v) for k, v in globals().items() if isinstance(v, dict)]:
    code = compile('def register_{n}(key, module): register(key, module, {n}s)'.format(n=k.rstrip('s')), __file__,
                   'exec')
    exec(code)

CODE_GEN_T = """
import fedflow.{pkg}.{sub_pkg}.{module} as {module}
import fedflow.register as register

ms = [v for k, v in inspect.getmembers({module}) if inspect.isclass(v)]
for m in ms:
    register.register_{sub_pkg}(module[: -len('_{{}}_handler'.format(sub_pkg))], m)
"""


def register_module(sub_pkg, modules):
    for module in modules:
        # __import__("fedflow.llm.model." + core_module)
        code = compile(CODE_GEN_T.format(pkg='llm', sub_pkg=sub_pkg, module=module), __file__, 'exec')
        exec(code)


print(inspect)
