import importlib.util
import os
import platform
import sys
from pathlib import Path

def load_pyo3_intree_native_module(crate_name: str, module_name: str):
    os_family = platform.system()
    if os_family == 'Linux':
        module_dylib_name = f'lib{crate_name}.so'
    elif os_family == 'Darwin':
        module_dylib_name = f'lib{crate_name}.dylib'
    elif os_family == 'Windows':
        module_dylib_name = f'{crate_name}.dll'
    else:
        raise ImportError('unknown os family: ' + os_family)

    output_path = 'release'
    if os.environ.get('PYO3_IMPORT_DEBUG') == '1':
        output_path = 'debug'

    module_path = Path(__file__).parent.parent / 'target' / output_path / module_dylib_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f'failed to locate native module for {module_name}, tried {module_path}')
    if spec.loader is None:
        raise ImportError('module loader missing?')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
