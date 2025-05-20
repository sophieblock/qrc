# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import importlib
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE



binaries = [
    ("/Users/so714f/opt/anaconda3/envs/wf_env/lib/python3.12/site-packages/z3/lib/libz3.dylib", "libz3.dylib")
]

block_cipher = None

packages = [
    'shiny',     'pyvis',    'jsonpickle',  'quimb',
    'toolz',     'autoray',  'cotengra',     'qiskit',
    'symengine', 'rustworkx','stevedore',   'attrs',
    'attr',      'visast',   'EoN',         'openfermion',
    'h5py',      'cirq',     'duet',         'mpl_toolkits',
    'pennylane', 'autograd', 'toml',         'cachetools',
    'faicons', 'olsq', 'fxpmath','qiskit'
]
datas = []
hiddenimports = [
    'shiny._launchbrowser',

    ]
hiddenimports += [
    'qiskit.transpiler.preset_passmanagers.builtin_plugins',
    'qiskit.transpiler.preset_passmanagers',
    'qiskit.transpiler.preset_passmanagers.plugin',
]

for pkg in packages:
    importlib.import_module(pkg)
    datas += collect_data_files(pkg)
    hiddenimports += collect_submodules(pkg)


WORKFLOW_PATH = os.environ.get("WORKFLOW_PATH", "/Users/so714f/Documents/code/workflow")



datas += [
    ('./app/run_app.py', './'),
    ('./app/app.py', './'),
    (WORKFLOW_PATH, './'),  # Reference the workflow directory directly
]


a = Analysis(
    ['app/run_app.py'],
    pathex=[os.path.abspath('.'), WORKFLOW_PATH],  # Include the workflow path
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    name='qrew_app',
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
