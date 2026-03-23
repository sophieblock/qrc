# Boeing CA Bundle: Command Reference Card

## Installation (Copy & Paste)

```bash
# Create bin directory
mkdir -p ~/bin ~/.config

# Navigate to workspace outputs (or wherever you saved files)
cd ~/.alter/outputs  # or wherever files are

# Copy scripts to bin and make executable
cp install-boeing-certs.sh ~/bin/
cp update-boeing-certs.sh ~/bin/
chmod +x ~/bin/install-boeing-certs.sh ~/bin/update-boeing-certs.sh

# Run main installation
~/bin/install-boeing-certs.sh

# When prompted, answer yes to:
# - Append Boeing certs to certifi (optional but recommended)
# - Import root to macOS Keychain (optional, adds system-wide trust)
```

## Fix %20 Filename Issue (Your Files)

```bash
cd ~/.ssl/certs

# Rename downloaded files
mv "Boeing%20BAS%20Issuing%20CA%20SHA256%20G4.crt" "Boeing_BAS_Issuing_CA_G4.crt"
mv "Boeing%20BAS%20Root%20CA%20SHA256%20G3.crt" "Boeing_BAS_Root_CA_G3.crt"
mv "Boeing%20Basic%20Assurance%20Software%20Issuing%20CA%20G3.crt" "Boeing_BAS_Software_Issuing_G3.crt"
mv "Boeing%20Basic%20Assurance%20Software%20Root%20CA%20G2.crt" "Boeing_BAS_Software_Root_G2.crt"

# Rebuild bundle
cp boeing-ca-bundle.pem boeing-ca-bundle.pem.bak
cat Boeing_BAS_Issuing_CA_G4.crt Boeing_BAS_Root_CA_G3.crt Boeing_BAS_Software_Issuing_G3.crt Boeing_BAS_Software_Root_G2.crt > boeing-ca-bundle.pem

# Verify
openssl crl2pkcs7 -nocrl -certfile boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout | head
```

## Verify Everything Works

```bash
# 1. Reload shell
source ~/.zshenv

# 2. Check environment variables
echo "SSL_CERT_FILE: $SSL_CERT_FILE"
echo "REQUESTS_CA_BUNDLE: $REQUESTS_CA_BUNDLE"

# 3. Test curl
curl -v https://sres.web.boeing.com/ 2>&1 | grep -i "certificate verify"

# 4. Test pip (should work without --trusted-host)
pip3 install --dry-run poetry

# 5. Test conda
conda search numpy

# 6. Show bundle contents
openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout -text | grep -E "Subject:|Not After" | head -20
```

## Check Certificate Expiration

```bash
# Show expiration dates
openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout -text | grep "Not After"

# Or use the update script
~/bin/update-boeing-certs.sh --check-only
```

## Configure Python/pip/Conda

```bash
# Add to ~/.zshenv (if install script didn't already)
echo 'export SSL_CERT_FILE="${HOME}/.ssl/certs/boeing-ca-bundle.pem"' >> ~/.zshenv
echo 'export REQUESTS_CA_BUNDLE="${HOME}/.ssl/certs/boeing-ca-bundle.pem"' >> ~/.zshenv

# Create pip config
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf <<EOF
[global]
cert = ${HOME}/.ssl/certs/boeing-ca-bundle.pem
EOF

# Create conda config
cat > ~/.condarc <<EOF
ssl_verify: ${HOME}/.ssl/certs/boeing-ca-bundle.pem
EOF

# Reload
source ~/.zshenv
```

## Set Up Automatic Weekly Updates (macOS)

```bash
# Copy launchd plist
cp local.boeing-ca-update.plist ~/Library/LaunchAgents/

# Load it (runs every Monday 2 AM)
launchctl load ~/Library/LaunchAgents/local.boeing-ca-update.plist

# Verify it's loaded
launchctl list | grep boeing-ca-update

# Check logs
tail -f /var/log/boeing-ca-update.log
```

## Manual Update/Verification

```bash
# Check status only (don't update)
~/bin/update-boeing-certs.sh --check-only

# Check and auto-update if needed
~/bin/update-boeing-certs.sh

# Force refresh
~/bin/update-boeing-certs.sh --force
```

## Handle Multiple OpenSSL Versions

```bash
# Find all openssl installations
which openssl                              # System
brew list --formula | grep openssl         # Homebrew
ls /usr/local/opt/openssl*/bin/openssl    # Intel Homebrew
ls /opt/homebrew/opt/openssl*/bin/openssl # Apple Silicon Homebrew

# Use specific openssl for install script
OPENSSL_BIN="/usr/local/opt/openssl@3/bin/openssl" ~/bin/install-boeing-certs.sh

# Control which openssl is in PATH (add to ~/.zshenv)
export PATH="/usr/local/opt/openssl@3/bin:$PATH"  # Prefer Homebrew openssl@3 on Intel
export PATH="/opt/homebrew/opt/openssl@3/bin:$PATH" # Prefer Homebrew openssl@3 on Apple Silicon
export PATH="/usr/bin:$PATH"                        # Prefer system openssl
```

## Troubleshooting

```bash
# Problem: certificate verify failed
# Solution: Check SSL_CERT_FILE and verify bundle
source ~/.zshenv
echo $SSL_CERT_FILE
ls -la $SSL_CERT_FILE
openssl crl2pkcs7 -nocrl -certfile $SSL_CERT_FILE | openssl pkcs7 -print_certs -noout > /dev/null && echo OK

# Problem: pip still says certificate error
# Solution: Test with explicit cert path
pip3 install --cert ~/.ssl/certs/boeing-ca-bundle.pem poetry

# Problem: conda not finding certs
# Solution: Verify .condarc
cat ~/.condarc
# Should contain: ssl_verify: /Users/so714f/.ssl/certs/boeing-ca-bundle.pem

# Problem: launchd job not running
# Solution: Check if it's loaded
launchctl list | grep boeing-ca-update
launchctl load ~/Library/LaunchAgents/local.boeing-ca-update.plist  # Reload if needed

# Problem: Wrong openssl being used
# Solution: Check and fix PATH
which openssl
openssl version
# Edit ~/.zshenv to adjust PATH order
```

## Restore from Backup

```bash
# List backups
ls -la ~/.ssl/certs/backups/

# Restore specific backup
cp ~/.ssl/certs/backups/20260323T044915Z/boeing-ca-bundle.pem.bak ~/.ssl/certs/boeing-ca-bundle.pem

# Verify restored bundle
openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout > /dev/null && echo OK
```

## File Locations Reference

```
~/.ssl/certs/
  └─ boeing-ca-bundle.pem           Canonical CA bundle
  └─ backups/
     └─ [timestamp]/
        └─ boeing-ca-bundle.pem.bak  Timestamped backups

~/.config/pip/
  └─ pip.conf                        Pip configuration

~/.condarc                            Conda SSL configuration
~/.zshenv                             Shell environment variables
~/.zshrc                              Shell rc (don't put SSL vars here)

~/bin/
  └─ install-boeing-certs.sh
  └─ update-boeing-certs.sh

~/Library/LaunchAgents/
  └─ local.boeing-ca-update.plist    Launchd job

/var/log/
  └─ boeing-ca-update.log            Launchd logs
  └─ boeing-ca-update.err
```

## Useful One-Liners

```bash
# Count certificates in bundle
grep -c "BEGIN CERTIFICATE" ~/.ssl/certs/boeing-ca-bundle.pem

# Show all certificate subjects
openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout -text | grep "Subject:"

# Test TLS without curl/openssl
python3 -c "import ssl,socket; ctx=ssl.create_default_context(cafile='$HOME/.ssl/certs/boeing-ca-bundle.pem'); s=ctx.wrap_socket(socket.socket(),server_hostname='sres.web.boeing.com'); s.connect(('sres.web.boeing.com',443)); print('ok'); s.close()"

# Check when certs expire
for f in ~/.ssl/certs/*.crt; do echo "$f:"; openssl x509 -noout -enddate -in "$f"; done

# View certificate chain for a host
openssl s_client -showcerts -connect sres.web.boeing.com:443 </dev/null

# Extract issuing CA from bundle
awk 'NR==1,/-----END CERTIFICATE-----/{print; exit}' ~/.ssl/certs/boeing-ca-bundle.pem

# Extract root CA from bundle (last cert)
awk '/-----BEGIN CERTIFICATE-----/{i++; flag=1; next} flag{print} /-----END CERTIFICATE-----/ && i==2{print; exit}' ~/.ssl/certs/boeing-ca-bundle.pem
```

## Environment Variable Reference

| Variable | Value | Used By | Where to Set |
|----------|-------|---------|---------------|
| SSL_CERT_FILE | /Users/so714f/.ssl/certs/boeing-ca-bundle.pem | openssl, curl, Python ssl | ~/.zshenv |
| REQUESTS_CA_BUNDLE | /Users/so714f/.ssl/certs/boeing-ca-bundle.pem | requests, pip | ~/.zshenv |
| OPENSSL_BIN | /path/to/openssl | install/update scripts | ~/.zshenv (optional) |
| CONDA_SSL_VERIFY | /path/to/cert (or false) | conda | ~/.zshenv (optional) |

## Which Shell File for What

| Content Type | File | Why |
|--------------|------|-----|
| CA/SSL paths (SSL_CERT_FILE, REQUESTS_CA_BUNDLE) | **~/.zshenv** | All shells, subprocesses, GUI apps |
| openssl/python/conda paths | **~/.zshenv** | Same reason |
| Aliases (alias ll=...) | ~/.zshrc | Interactive shells only, safe |
| Functions/completions | ~/.zshrc | Interactive shells only, safe |
| PATH modifications | **~/.zshenv** | All shells (scripts need correct PATH) |
| Login-only setup | ~/.zprofile | Login shells only (rare on macOS) |

---

**Quick Decision Tree:**
1. First time setup → Run install-boeing-certs.sh
2. Certificate expired/issues → Run update-boeing-certs.sh --force
3. Manual check → Run update-boeing-certs.sh --check-only
4. Auto weekly checks → launchctl load ~/Library/LaunchAgents/local.boeing-ca-update.plist
5. Test specific tool → Use commands from "Verify Everything Works" section
