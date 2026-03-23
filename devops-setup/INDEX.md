# Boeing CA Bundle Setup Toolkit – Complete Index

## 📋 What You Have

A complete, production-ready solution for managing Boeing CA certificates on macOS without disabling TLS verification.

### Files in This Package

```
.
├── 📖 README.md                          Full technical documentation (398 lines)
├── 🚀 QUICKSTART-SOPHIE.md              Personalized guide for your situation (330 lines)
├── 📝 COMMAND-REFERENCE.md              Copy-paste command reference (230+ lines)
├── 📄 INDEX.md                           This file
├── 🔧 install-boeing-certs.sh           Main installation script (306 lines)
├── 🔄 update-boeing-certs.sh            Verification & update script (116 lines)
└── ⏰ local.boeing-ca-update.plist      macOS launchd job for weekly checks
```

## 🎯 Where to Start

### If You're in a Hurry (5 min)
→ Read **COMMAND-REFERENCE.md** and copy the "Installation" commands

### If This Is Your First Time
→ Read **QUICKSTART-SOPHIE.md** first, then run install-boeing-certs.sh

### If You Need Full Context
→ Read **README.md** for complete documentation

### If You Need to Fix Your %20 Filenames
→ Go to **QUICKSTART-SOPHIE.md** → "Fix the Downloaded Files (%20 issue)"

## 📦 Quick Installation

```bash
mkdir -p ~/bin
cp install-boeing-certs.sh ~/bin/
cp update-boeing-certs.sh ~/bin/
chmod +x ~/bin/*.sh

# Run the installer
~/bin/install-boeing-certs.sh

# Verify
source ~/.zshenv
pip3 install poetry
conda search numpy
```

Done. No --trusted-host, proper TLS verification, no security workarounds.

## 📚 File Guide

### README.md
**Purpose:** Complete technical reference  
**Length:** 398 lines  
**Contains:**
- Full overview of all components
- Detailed step-by-step walkthrough
- OpenSSL/Homebrew handling
- Environment variable reference (which file to edit)
- pip, conda, curl configuration
- Verification commands
- Troubleshooting guide (organized by problem)
- Backup & recovery procedures
- Maintenance checklist

**When to read:** Once, to understand the whole system

---

### QUICKSTART-SOPHIE.md  
**Purpose:** Personalized guide for YOUR specific situation  
**Length:** 330 lines  
**Addresses:**
- Your %20 filename issue (specific rename commands)
- Your existing ~/.ssl/certs/boeing-ca-bundle.pem file
- Your Boeing Keychain certificates (G2/G3/G4 from keychain screenshot)
- Transitioning from --trusted-host to proper TLS
- Conda-specific setup
- Multiple openssl handling

**When to read:** First, before running install script

---

### COMMAND-REFERENCE.md  
**Purpose:** Copy-paste commands for common tasks  
**Length:** 230+ lines  
**Sections:**
- Installation (ready to copy)
- Fix %20 filename issue
- Verify everything works
- Check certificate expiration
- Configure Python/pip/conda
- Set up automatic weekly updates
- Troubleshooting (one-liners)
- Restore from backup
- File location reference
- Useful command recipes
- Environment variable matrix
- Which shell file for what

**When to use:** Daily reference, copy commands as needed

---

### install-boeing-certs.sh  
**Purpose:** Initial setup and configuration  
**Type:** Executable bash script  
**Lines:** 306  
**Does:**
1. Downloads Boeing CA certs (no %20 encoding issues)
2. Converts DER→PEM if needed
3. Builds canonical PEM bundle
4. Verifies bundle contents
5. Tests TLS connections
6. Configures SSL_CERT_FILE & REQUESTS_CA_BUNDLE in ~/.zshenv
7. Configures pip (~/.config/pip/pip.conf)
8. Configures conda (~/.condarc)
9. Optionally appends to certifi
10. Optionally imports root to macOS System Keychain

**Usage:**
```bash
~/bin/install-boeing-certs.sh                    # Interactive (prompts for optional steps)
NO_PROMPT=1 ~/bin/install-boeing-certs.sh       # Non-interactive
OPENSSL_BIN=/path/to/openssl ~/bin/install-boeing-certs.sh  # Use specific openssl
```

**When to run:** First-time setup or major CA rollout

---

### update-boeing-certs.sh  
**Purpose:** Periodic verification & auto-update  
**Type:** Executable bash script  
**Lines:** 116  
**Does:**
1. Checks certificate expiration dates
2. Warns if < 30 days to expiry
3. Tests TLS to sres.web.boeing.com and git.web.boeing.com
4. Auto-updates if any cert is expired (optional)
5. Creates timestamped backups

**Usage:**
```bash
~/bin/update-boeing-certs.sh                 # Auto-update if issues found
~/bin/update-boeing-certs.sh --check-only   # Check status only, don't update
~/bin/update-boeing-certs.sh --force         # Force refresh
```

**When to run:** Weekly (manual) or daily (via launchd)

---

### local.boeing-ca-update.plist  
**Purpose:** macOS launchd job for automatic weekly verification  
**Type:** XML property list  
**Lines:** 34  
**Schedule:** Every Monday at 2:00 AM  
**Logs:** /var/log/boeing-ca-update.{log,err}  

**Installation:**
```bash
cp local.boeing-ca-update.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/local.boeing-ca-update.plist
```

**When to use:** For hands-off automatic verification (recommended)

---

## 🔑 Key Concepts Explained

### Environment Variables (Why and Where)

**SSL_CERT_FILE & REQUESTS_CA_BUNDLE**
- **What:** Path to your Boeing CA bundle
- **Used by:** openssl, curl, Python ssl module, requests, pip
- **Where:** ~/.zshenv (NOT ~/.zshrc or ~/.zprofile)
- **Why ~/.zshenv?** It's sourced for ALL zsh invocations (login, non-login, scripts, GUI apps)

### Configuration Files (Who Needs What)

| Tool | Config File | Setting |
|------|-------------|----------|
| Python/openssl/curl | ~/.zshenv env var | SSL_CERT_FILE |
| pip | ~/.config/pip/pip.conf | cert = /path/to/bundle.pem |
| conda | ~/.condarc | ssl_verify: /path/to/bundle.pem |
| All requests | ~/.zshenv env var | REQUESTS_CA_BUNDLE |

### Multiple OpenSSL Versions

If you have both system and Homebrew openssl, scripts can use either:

```bash
# Use Homebrew openssl@3 on Intel
OPENSSL_BIN="/usr/local/opt/openssl@3/bin/openssl" ~/bin/install-boeing-certs.sh

# Use Homebrew openssl@3 on Apple Silicon
OPENSSL_BIN="/opt/homebrew/opt/openssl@3/bin/openssl" ~/bin/install-boeing-certs.sh

# Use system openssl (default)
~/bin/install-boeing-certs.sh
```

You cannot prevent Homebrew from installing duplicate dependencies, but you can control which is used.

## ✅ Verification Checklist

After running install-boeing-certs.sh:

- [ ] Bundle file exists: `ls -la ~/.ssl/certs/boeing-ca-bundle.pem`
- [ ] Environment variables set: `echo $SSL_CERT_FILE`
- [ ] Curl works: `curl -v https://sres.web.boeing.com/ 2>&1 | grep certificate`
- [ ] pip works: `pip3 install --dry-run poetry`
- [ ] conda works: `conda search numpy`
- [ ] Bundle has certs: `grep -c "BEGIN CERTIFICATE" ~/.ssl/certs/boeing-ca-bundle.pem`
- [ ] pip.conf exists: `cat ~/.config/pip/pip.conf`
- [ ] .condarc exists: `cat ~/.condarc`
- [ ] .zshenv updated: `grep SSL_CERT_FILE ~/.zshenv`

## 🆘 Troubleshooting Quick Links

| Problem | Solution Location |
|---------|-------------------|
| %20 in filenames | QUICKSTART-SOPHIE.md → "Fix the Downloaded Files" |
| "certificate verify failed" | README.md → Troubleshooting → "certificate verify failed" |
| pip/conda still failing | COMMAND-REFERENCE.md → Troubleshooting section |
| Multiple openssl issues | README.md → "Multiple OpenSSL Installations" |
| launchd not running | README.md → Troubleshooting → "launchd job not running" |
| Need to restore backup | COMMAND-REFERENCE.md → "Restore from Backup" |
| Certificate expired | Run `~/bin/update-boeing-certs.sh --force` |

## 📞 Support Resources

- **Boeing PKI Information:** https://boeing.sharepoint.us/sites/IAMPKI
- **PKI Office Hours:** Tuesdays & Thursdays (check SharePoint)
- **Enterprise Help Desk:** For certificate issues and requests

## 🚀 Next Steps

1. **Pick your starting point** (see "Where to Start" above)
2. **Copy the toolkit files** from .alter/outputs to your machine
3. **Run install-boeing-certs.sh** (or follow QUICKSTART-SOPHIE.md step-by-step)
4. **Verify with the commands** in "Verification Checklist" above
5. **(Optional) Set up automatic updates** via launchd
6. **Keep the documentation** for reference and troubleshooting

## 📋 Summary

**What you get:**
- ✅ Proper TLS verification (no --trusted-host workarounds)
- ✅ Automatic CA certificate management
- ✅ Works with Python, pip, conda, curl, git
- ✅ Optional automatic weekly verification
- ✅ Backup & recovery procedures
- ✅ Multiple OpenSSL version support
- ✅ Complete documentation & troubleshooting

**Time to implement:**
- 5 minutes: Initial setup
- 5 minutes: Verification
- 2 minutes: Optional launchd setup
- **Total: ~12 minutes for complete, long-term solution**

**Files you'll modify:**
- ~/.zshenv (add SSL_CERT_FILE, REQUESTS_CA_BUNDLE)
- ~/.config/pip/pip.conf (create if needed)
- ~/.condarc (create if needed)
- ~/.ssl/certs/boeing-ca-bundle.pem (rebuild from fresh certs)

All changes are non-invasive, easily reversible via backups.

---

**Start with:** QUICKSTART-SOPHIE.md or install-boeing-certs.sh

**Reference:** README.md for complete details, COMMAND-REFERENCE.md for quick commands
