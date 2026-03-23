# File Structure & Directory Guide

## After Installation, Your Directories Will Look Like This

```
~/ (home directory)
├── .ssl/
│   └── certs/
│       ├── boeing-ca-bundle.pem                ← Canonical bundle (the important file)
│       ├── Boeing_BAS_Issuing_CA_G4.crt        ← Individual certs
│       ├── Boeing_BAS_Root_CA_G3.crt           ← (optional, kept for reference)
│       ├── boeing-ca-bundle.pem.old-20260323   ← Pre-G3/G4 backup (example date)
│       └── backups/
│           ├── 20260323T045000Z/               ← Timestamped backup folders
│           │   ├── boeing-ca-bundle.pem.bak    ← Point-in-time backup
│           │   └── certifi/
│           │       └── cacert.pem.bak          ← Python certifi backup (if appended)
│           └── 20260316T030000Z/               ← Older backup
│               └── boeing-ca-bundle.pem.bak
│
├── .config/
│   └── pip/
│       └── pip.conf                            ← pip configuration (cert path)
│
├── .zshenv                                      ← Shell environment (SSL_CERT_FILE, etc.)
├── .condarc                                     ← conda configuration (ssl_verify)
├── .zshrc                                       ← Shell rc (interactive shells, no SSL vars)
│
├── bin/
│   ├── install-boeing-certs.sh                 ← Main installation script
│   └── update-boeing-certs.sh                  ← Periodic verification script
│
└── Library/
    └── LaunchAgents/
        └── local.boeing-ca-update.plist        ← macOS launchd job (weekly)

/var/log/ (system logs)
├── boeing-ca-update.log                        ← launchd task logs (if using automation)
└── boeing-ca-update.err
```
