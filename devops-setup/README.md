# Boeing CA Bundle Setup for macOS/Linux

## Overview

This toolkit installs and maintains Boeing root & issuing CA certificates for secure TLS connections to Boeing infrastructure (Artifactory, Git, etc.) without disabling TLS verification.

## Files Included

1. **install-boeing-certs.sh** – Initial setup; downloads, verifies, and configures bundle
2. **update-boeing-certs.sh** – Periodic verification and update (like Linux `update-ca-certificates`)
3. **local.boeing-ca-update.plist** – macOS launchd config for weekly automated checks
4. **README.md** – This file

## Quick Start (5 minutes)

### 1. Download and Install

```bash
# Create bin directory if it doesn't exist
mkdir -p ~/bin

# Copy install script (make it executable)
cp install-boeing-certs.sh ~/bin/
chmod +x ~/bin/install-boeing-certs.sh

# Also copy update script
cp update-boeing-certs.sh ~/bin/
chmod +x ~/bin/update-boeing-certs.sh

# Run the installer
~/bin/install-boeing-certs.sh
```

### 2. Verify Installation

```bash
# Reload shell environment
source ~/.zshenv

# Test Python/pip
python3 -c "import ssl,os; print('SSL_CERT_FILE:', os.environ.get('SSL_CERT_FILE'))"

# Test curl
curl -v https://sres.web.boeing.com/ 2>&1 | grep -i "certificate verify ok"

# Test pip (dry-run, no actual install)
pip3 install --dry-run poetry
```

### 3. (Optional) Set Up Automatic Weekly Updates

For macOS only:

```bash
# Copy the launchd plist
cp local.boeing-ca-update.plist ~/Library/LaunchAgents/

# Load it (runs every Monday at 2:00 AM)
launchctl load ~/Library/LaunchAgents/local.boeing-ca-update.plist

# Verify it's loaded
launchctl list | grep boeing-ca-update
```

For Linux, add to crontab:

```bash
crontab -e
# Add line:
0 2 * * 1 source ~/.zshenv && ${HOME}/bin/update-boeing-certs.sh --check-only >> /var/log/boeing-ca-update.log 2>&1
```

## What Each Script Does

### install-boeing-certs.sh

**Purpose:** Initial setup and bundle creation

**Steps:**
1. Creates ~/.ssl/certs/ directory and backups
2. Downloads Boeing CA certificates from Boeing CRL server
3. Converts DER → PEM if needed
4. Builds canonical bundle at ~/.ssl/certs/boeing-ca-bundle.pem
5. Verifies bundle contents and TLS connections
6. Configures:
   - ~/.zshenv (SSL_CERT_FILE, REQUESTS_CA_BUNDLE env vars)
   - ~/.config/pip/pip.conf (cert path for pip)
   - ~/.condarc (ssl_verify for conda)
7. (Optional) Appends to Python certifi bundle
8. (Optional for macOS) Imports root CA to System Keychain

**Usage:**
```bash
~/bin/install-boeing-certs.sh                    # Interactive (prompts for optional steps)
NO_PROMPT=1 ~/bin/install-boeing-certs.sh       # Non-interactive
OPENSSL_BIN=/usr/local/opt/openssl@3/bin/openssl ~/bin/install-boeing-certs.sh  # Force specific openssl
```

### update-boeing-certs.sh

**Purpose:** Periodic verification; checks expiration, tests TLS, and auto-updates if needed

**Usage:**
```bash
~/bin/update-boeing-certs.sh                     # Check status; auto-update if issues found
~/bin/update-boeing-certs.sh --check-only       # Check status only, don't update
~/bin/update-boeing-certs.sh --force             # Force refresh even if current
```

**What it checks:**
- Certificate expiration dates
- Days until expiry (warns if < 30 days)
- TLS connections to sres.web.boeing.com and git.web.boeing.com
- Auto-updates bundle if any certificate is expired

### local.boeing-ca-update.plist

**Purpose:** macOS launchd job that runs weekly verification

**Installation:**
```bash
cp local.boeing-ca-update.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/local.boeing-ca-update.plist
```

**Schedule:** Every Monday at 2:00 AM (adjust by editing StartCalendarInterval in the plist)

**Logs:** /var/log/boeing-ca-update.log and .err

## Multiple OpenSSL Installations

If Homebrew or other package managers installed their own openssl (common), you can explicitly specify which one to use:

### Find available openssl versions

```bash
which openssl                                    # System openssl
brew list --formula | grep openssl              # Homebrew openssl
ls /usr/local/opt/openssl*/bin/openssl         # Homebrew installs
ls /opt/homebrew/opt/openssl*/bin/openssl      # Apple Silicon Homebrew
```

### Use a specific openssl

```bash
# For Homebrew openssl@3 on Intel
OPENSSL_BIN="/usr/local/opt/openssl@3/bin/openssl" ~/bin/install-boeing-certs.sh

# For Homebrew openssl@3 on Apple Silicon
OPENSSL_BIN="/opt/homebrew/opt/openssl@3/bin/openssl" ~/bin/install-boeing-certs.sh

# System openssl (default)
OPENSSL_BIN="/usr/bin/openssl" ~/bin/install-boeing-certs.sh
```

### Prevent Homebrew from installing duplicate openssl

Unfortunately, you cannot fully prevent a dependency from being installed. However, you can:

1. **Use system packages where possible:**
   - macOS comes with OpenSSL via libressl
   - Use system python3, git, curl (don't reinstall via brew if not needed)

2. **Manage with PATH:**
   - Control which openssl is used by adjusting ~/.zshenv:
     ```bash
     export PATH="/usr/local/opt/openssl@3/bin:$PATH"  # Use Homebrew openssl
     export PATH="/usr/bin:$PATH"                       # Prefer system openssl (put first)
     ```

3. **Use Python virtual environments:**
   - Isolate Python packages and their dependencies:
     ```bash
     python3 -m venv myenv
     source myenv/bin/activate
     pip install poetry  # installs into venv, not system
     ```

4. **Track explicitly:**
   - Audit Homebrew dependencies periodically:
     ```bash
     brew deps --tree openssl  # see what depends on openssl
     brew leaves              # see top-level installed packages
     ```

## Environment Variables (zshenv, zshrc, zprofile)

Where to set SSL/CA variables:

| Variable | Purpose | File | Why |
|----------|---------|------|-----|
| SSL_CERT_FILE | Path to CA bundle for openssl/curl/Python | **~/.zshenv** | Available to all shells & subprocesses (login & non-login) |
| REQUESTS_CA_BUNDLE | Path to CA bundle for requests/pip | **~/.zshenv** | Same reason |
| OPENSSL_BIN | Override which openssl binary to use (optional) | **~/.zshenv** | Makes it available to scripts |
| PATH | Control openssl precedence (optional) | **~/.zshenv** or ~/.zshrc | If using .zshrc, only interactive shells see it |

**Key differences:**
- **.zshenv** – sourced for ALL zsh invocations (login, non-login, scripts, GUI apps if spawned after login)
- **.zshrc** – sourced ONLY for interactive login shells (safe for aliases/functions, not guaranteed for scripts)
- **.zshprofile** – sourced ONLY on macOS for login shells; generally avoid unless you have specific needs

## Configuration Files (permanent settings)

These are created/updated by install-boeing-certs.sh:

### ~/.config/pip/pip.conf
```ini
[global]
cert = /Users/so714f/.ssl/certs/boeing-ca-bundle.pem
```
**Effect:** pip always uses this cert (even if SSL_CERT_FILE is unset)

### ~/.condarc
```yaml
ssl_verify: /Users/so714f/.ssl/certs/boeing-ca-bundle.pem
```
**Effect:** conda respects this cert for package verification

### Python certifi (optional append)
If you chose to append during install-boeing-certs.sh, the Boeing certs are appended to:
```
$(python3 -c "import certifi; print(certifi.where())")
```
Backing up before modification. Drawback: certifi upgrades may overwrite.

## Verification Commands

Run these after installation to confirm everything works:

```bash
# 1. Check environment variables
echo $SSL_CERT_FILE
echo $REQUESTS_CA_BUNDLE

# 2. Verify Python sees the bundle
python3 -c "import ssl,os; print('SSL_CERT_FILE:', os.environ.get('SSL_CERT_FILE')); paths=ssl.get_default_verify_paths(); print('Python paths:', paths)"

# 3. Test curl (should show "certificate verify ok")
curl -v https://sres.web.boeing.com/ 2>&1 | grep -i "certificate verify"

# 4. Test pip (dry-run, no actual install)
pip3 install --dry-run poetry 2>&1 | head -20

# 5. Test conda
conda search numpy 2>&1 | head -5

# 6. Show bundle contents
openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout -text | grep -E "Subject:|Issuer:|Not After" | head -20

# 7. Check which openssl is being used
which openssl
openssl version

# 8. Test TLS handshake with bundle
openssl s_client -CAfile ~/.ssl/certs/boeing-ca-bundle.pem -connect sres.web.boeing.com:443 </dev/null
```

## Troubleshooting

### "certificate verify failed"

1. Check SSL_CERT_FILE is set:
   ```bash
   echo $SSL_CERT_FILE
   source ~/.zshenv && echo $SSL_CERT_FILE
   ```

2. Verify bundle exists and has certs:
   ```bash
   ls -la ~/.ssl/certs/boeing-ca-bundle.pem
   grep "BEGIN CERTIFICATE" ~/.ssl/certs/boeing-ca-bundle.pem | wc -l
   ```

3. Test with explicit bundle path:
   ```bash
   curl --cacert ~/.ssl/certs/boeing-ca-bundle.pem -v https://sres.web.boeing.com/
   python3 -c "import ssl,socket; ctx=ssl.create_default_context(cafile='$HOME/.ssl/certs/boeing-ca-bundle.pem'); s=ctx.wrap_socket(socket.socket(),server_hostname='sres.web.boeing.com'); s.connect(('sres.web.boeing.com',443)); print('ok'); s.close()"
   ```

4. Check certificate expiration:
   ```bash
   openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout -text | grep "Not After"
   ```
   If expired, run: `~/bin/update-boeing-certs.sh --force`

### Different openssl being used

1. Check which openssl is in PATH:
   ```bash
   which openssl
   openssl version
   ```

2. If wrong version, adjust PATH in ~/.zshenv:
   ```bash
   export PATH="/usr/local/opt/openssl@3/bin:$PATH"  # Homebrew openssl first
   export PATH="/usr/bin:$PATH"                       # System openssl first (macOS default)
   ```

3. Force specific openssl for scripts:
   ```bash
   OPENSSL_BIN="/usr/local/opt/openssl@3/bin/openssl" ~/bin/update-boeing-certs.sh
   ```

### pip still using --trusted-host workaround

1. Remove workaround from pip config:
   ```bash
   cat ~/.config/pip/pip.conf  # remove any --trusted-host entries
   ```

2. Test with the cert file instead:
   ```bash
   pip3 install --cert ~/.ssl/certs/boeing-ca-bundle.pem poetry
   ```

3. If still failing, verify cert bundle:
   ```bash
   openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout >/dev/null && echo "Bundle OK" || echo "Bundle corrupt"
   ```

### launchd job not running

1. Check if loaded:
   ```bash
   launchctl list | grep boeing-ca-update
   ```

2. If not listed, reload:
   ```bash
   launchctl load ~/Library/LaunchAgents/local.boeing-ca-update.plist
   ```

3. Check logs:
   ```bash
   tail -f /var/log/boeing-ca-update.log
   tail -f /var/log/boeing-ca-update.err
   ```

4. Manual test (run the plist command directly):
   ```bash
   /bin/bash -c "source ~/.zshenv && ${HOME}/bin/update-boeing-certs.sh --check-only"
   ```

## Backup & Recovery

All backups are created in ~/.ssl/certs/backups/:

```bash
# List all backups
ls -la ~/.ssl/certs/backups/

# Restore from specific backup date
cp ~/.ssl/certs/backups/20260323T044915Z/boeing-ca-bundle.pem.bak ~/.ssl/certs/boeing-ca-bundle.pem

# Verify restored bundle
openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout | head
```

## Conda-Specific Notes

After running install-boeing-certs.sh, conda will use the bundle automatically via ~/.condarc:

```bash
# Test conda channel access
conda search numpy

# If issues, verify condarc
cat ~/.condarc

# Force conda to use a specific channel or verify:
conda search numpy --channel defaults
```

For conda-build or environments with custom pip, make sure SSL_CERT_FILE is set:

```bash
source ~/.zshenv
conda run -n myenv pip install poetry  # inherits SSL_CERT_FILE from parent shell
```

## Support & Updates

- **Boeing PKI Information:** https://boeing.sharepoint.us/sites/IAMPKI
- **PKI Office Hours:** Tuesdays & Thursdays (check SharePoint for times)
- **Enterprise Help Desk:** For escalations and certificate request issues

## Maintenance Checklist

- [ ] Run install-boeing-certs.sh after Boeing PKI rollouts (new G3/G4 CAs announced)
- [ ] Check launchd logs monthly: `tail -f /var/log/boeing-ca-update.log`
- [ ] Verify bundle expiration quarterly: `update-boeing-certs.sh --check-only`
- [ ] Back up ~/.ssl/certs/ when changing security-critical configs
- [ ] Test pip/conda/curl after macOS major updates (sometimes resets cert paths)

## License & Attribution

These scripts are provided as-is for Boeing internal use. Adapt as needed for your organization.
