# -*- mode: python; coding: utf-8 -*-
import os
import importlib
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE

block_cipher = None

# 1) list every top-level package you need bundled
packages = [
    'shiny',     'pyvis',    'jsonpickle',  'quimb',
    'toolz',     'autoray',  'cotengra',     'qiskit',
    'symengine', 'rustworkx','stevedore',   'attrs',
    'attr',      'visast',   'EoN',         'openfermion',
    'h5py',      'cirq',     'duet',         'mpl_toolkits',
    'pennylane', 'autograd', 'toml',         'cachetools',
]

# 2) collect everything under those packages
datas = []
hiddenimports = ['shiny._launchbrowser']
binaries = []

for pkg in packages:
    # ensure module is importable
    importlib.import_module(pkg)

    # collect any data files (e.g. non-py files)
    datas += collect_data_files(pkg)

    # collect all submodules so nothing is missed
    hiddenimports += collect_submodules(pkg)

# 3) add your own application files
datas += [
    ('./app/run_app.py', '.'),    # your entrypoint
    ('./workflow',    'workflow'),
    ('./app/app.py',  'app'),
]

# 4) now build
a = Analysis(
    ['app/run_app.py'],
    pathex=[os.path.abspath('.')],  # include current dir on sys.path
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
    debug=False,
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
# -*- mode: python ; coding: utf-8 -*-


block_cipher = None
shiny = os.path.abspath("./venv/Lib/site-packages/shiny")
pyvis = os.path.abspath("./venv/Lib/site-packages/pyvis")
jsonpickle = os.path.abspath("./venv/Lib/site-packages/jsonpickle")
quimb = os.path.abspath("./venv/Lib/site-packages/quimb")
toolz = os.path.abspath("./venv/Lib/site-packages/toolz")
autoray = os.path.abspath("./venv/Lib/site-packages/autoray")
cotengra = os.path.abspath("./venv/Lib/site-packages/cotengra")
qiskit = os.path.abspath("./venv/Lib/site-packages/qiskit")
symengine = os.path.abspath("./venv/Lib/site-packages/symengine")
rustworkx = os.path.abspath("./venv/Lib/site-packages/rustworkx")
stevedore = os.path.abspath("./venv/Lib/site-packages/stevedore")
attrs = os.path.abspath("./venv/Lib/site-packages/attrs")
attr = os.path.abspath("./venv/Lib/site-packages/attr")
visast = os.path.abspath("./venv/Lib/site-packages/visast")
eon = os.path.abspath("./venv/Lib/site-packages/EoN")
openfermion = os.path.abspath("./venv/Lib/site-packages/openfermion")
h5py = os.path.abspath("./venv/Lib/site-packages/h5py")
cirq = os.path.abspath("./venv/Lib/site-packages/cirq")
duet = os.path.abspath("./venv/Lib/site-packages/duet")
mpl_toolkits = os.path.abspath("./venv/Lib/site-packages/mpl_toolkits")
pennylane = os.path.abspath("./venv/Lib/site-packages/pennylane")
autograd = os.path.abspath("./venv/Lib/site-packages/autograd")
toml = os.path.abspath("./venv/Lib/site-packages/toml")
cachetools = os.path.abspath("./venv/Lib/site-packages/cachetools")

a = Analysis(
    ['app/run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('./app/run_app.py', './'), ('./workflow', './'), ('./app/app.py', './'), (shiny,'./shiny'), (pyvis, './pyvis'), (jsonpickle, './jsonpickle'), (quimb, './quimb'), (toolz, './toolz'), (autoray, './autoray'), (cotengra, './cotengra'), (qiskit, './qiskit'), (symengine, './symengine'), (rustworkx, './rustworkx'), (stevedore, './stevedore'), (attrs, './attrs'), (attr, './attr'), (visast, './visast'), (eon, './EoN'), (openfermion, './openfermion'), (h5py, './h5py'), (cirq, './cirq'), (duet, './duet'), (mpl_toolkits, './mpl_toolkits'), (pennylane, './pennylane'), (autograd, './autograd'), (toml, './toml'), (cachetools, './cachetools')],
    hiddenimports=['shiny._launchbrowser'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
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
    debug=False,
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
