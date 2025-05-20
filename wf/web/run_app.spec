# -*- mode: python; coding: utf-8 -*-
import os, sys, importlib, subprocess
import importlib
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE
import qiskit

block_cipher = None
libz3 = "/Users/sophieblock/miniforge3/envs/torch_env/lib/python3.11/site-packages/z3/lib/libz3.dylib"

binaries = [(libz3, ".")]

packages = [
    'shiny',     'pyvis',    'jsonpickle',  'quimb',
    'toolz',     'autoray',  'cotengra',     'qiskit',
    'symengine', 'rustworkx','stevedore',   'attrs',
    'attr',      'visast',   'EoN',         'openfermion',
    'h5py',      'cirq',     'duet',         'mpl_toolkits',
    'pennylane', 'autograd', 'toml',         'cachetools',
    'faicons', 'fxpmath', 'qiskit'
]


datas = []
hiddenimports = ['shiny._launchbrowser',
'qiskit.transpiler.preset_passmanagers.builtin_plugins',]


for pkg in packages: 
    importlib.import_module(pkg)
    datas += collect_data_files(pkg)
    hiddenimports += collect_submodules(pkg)


# now bundle the dist-info
site_pkgs = os.path.dirname(os.path.dirname(qiskit.__file__))
dist_info = os.path.join(site_pkgs, f"qiskit-{qiskit.__version__}.dist-info")
datas.append((dist_info, os.path.basename(dist_info)))
WORKFLOW_PATH = os.environ.get("WORKFLOW_PATH", "/Users/sophieblock/torch_wf")


datas += [
    ('./app/run_app.py', './'),
    ('./app/app.py', './'),
    (WORKFLOW_PATH, './'), 
]


a = Analysis(
    ['app/run_app.py'],
    pathex=[os.path.abspath('.'), WORKFLOW_PATH], 
    
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hook-inspect.py', 'pkg_resources_qiskit_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    debug=True,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='qrew_app2',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)