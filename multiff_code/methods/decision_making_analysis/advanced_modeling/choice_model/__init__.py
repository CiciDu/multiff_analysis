# __init__.py
import os
import pkgutil
import importlib

__all__ = []  # list of things exported when doing `from myfolder import *`

# Iterate through all modules in this package
package_dir = os.path.dirname(__file__)
for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
    if not is_pkg:
        module = importlib.import_module(f"{__name__}.{module_name}")
        # Export all functions/vars that don’t start with "_"
        for attr_name in dir(module):
            if not attr_name.startswith("_"):
                globals()[attr_name] = getattr(module, attr_name)
                __all__.append(attr_name)

