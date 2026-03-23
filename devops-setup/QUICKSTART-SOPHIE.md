# Boeing CA Bundle: Quick Setup for so714f

## Your Situation

- You already have ~/.ssl/certs/boeing-ca-bundle.pem (from before G3/G4 rollout)
- Your macOS Keychain contains Boeing G2/G3/G4 certificates (from prior setup)
- Downloaded files have %20 encoding in names (URL-encoded spaces)
- pip works with --trusted-host but you want proper TLS verification
- Need to move to conda without security workarounds

## Immediate Actions (Today)

### 1. Fix the Downloaded Files (%20 issue)

The curl command downloads files with %20 in the filename. Rename them:

```bash
cd ~/.ssl/certs

# List files with %20
ls -la | grep "%20"

# Rename each (or use a loop)
mv "Boeing%20BAS%20Issuing%20CA%20SHA256%20G4.crt" "Boeing_BAS_Issuing_CA_G4.crt"
mv "Boeing%20BAS%20Root%20CA%20SHA256%20G3.crt" "Boeing_BAS_Root_CA_G3.crt"
mv "Boeing%20Basic%20Assurance%20Software%20Issuing%20CA%20G3.crt" "Boeing_BAS_Software_Issuing_G3.crt"
mv "Boeing%20Basic%20Assurance%20Software%20Root%20CA%20G2.crt" "Boeing_BAS_Software_Root_G2.crt"

# Verify
ls -la *.crt
```

### 2. Rebuild Your Bundle (Properly)

```bash
cd ~/.ssl/certs

# Backup your old bundle
cp boeing-ca-bundle.pem boeing-ca-bundle.pem.old-$(date +%Y%m%d)

# Create fresh bundle (order: Issuing CAs first, then Root)
cat Boeing_BAS_Issuing_CA_G4.crt Boeing_BAS_Software_Issuing_G3.crt Boeing_BAS_Root_CA_G3.crt Boeing_BAS_Software_Root_G2.crt > boeing-ca-bundle.pem

# Verify bundle contents
openssl crl2pkcs7 -nocrl -certfile boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout -text | grep -E "Subject:|Issuer:|Not After" | head -20
```

### 3. Verify Bundle Works

```bash
# Test with curl
curl --cacert ~/.ssl/certs/boeing-ca-bundle.pem -v https://sres.web.boeing.com/ 2>&1 | grep -i "certificate verify"

# Test with openssl
openssl s_client -CAfile ~/.ssl/certs/boeing-ca-bundle.pem -connect sres.web.boeing.com:443 </dev/null 2>&1 | grep "Verify return code"

# Both should show OK
```

### 4. Configure Environment (One Time)

Add to ~/.zshenv (create if doesn't exist):

```bash
# Add these lines
echo 'export SSL_CERT_FILE="${HOME}/.ssl/certs/boeing-ca-bundle.pem"' >> ~/.zshenv
echo 'export REQUESTS_CA_BUNDLE="${HOME}/.ssl/certs/boeing-ca-bundle.pem"' >> ~/.zshenv

# Reload
source ~/.zshenv

# Verify
echo $SSL_CERT_FILE
```

### 5. Configure pip (One Time)

```bash
mkdir -p ~/.config/pip

cat > ~/.config/pip/pip.conf <<EOF
[global]
cert = ${HOME}/.ssl/certs/boeing-ca-bundle.pem
EOF

# Verify
cat ~/.config/pip/pip.conf
```

### 6. Configure conda (One Time)

```bash
cat > ~/.condarc <<EOF
ssl_verify: ${HOME}/.ssl/certs/boeing-ca-bundle.pem
EOF

# Verify
cat ~/.condarc
```

## Test Everything

```bash
# Reload shell
source ~/.zshenv

# Test pip (no --trusted-host)
pip3 install poetry

# Test conda
conda search numpy

# Test curl
curl https://sres.web.boeing.com/ -v 2>&1 | grep -i "certificate verify ok"
```

## About Your Keychain

Your macOS Keychain already contains Boeing G2/G3/G4 certs (shown in your screenshot). This is fine and actually helpful:

- macOS system tools (curl, git, etc.) that use Keychain will trust these
- Python/pip/conda use SSL_CERT_FILE or certifi bundle, not Keychain directly
- Having both redundancy is secure

**No action needed on Keychain** unless you want system-wide trust (optional).

If you want to add the root CA to system-wide trust:

```bash
# Extract root CA from bundle
cd ~/.ssl/certs

# Get the last (root) certificate
awk 'BEGIN{i=0} /-----BEGIN CERTIFICATE-----/{i++} i==4' boeing-ca-bundle.pem > boeing_root.pem

# Import to System Keychain (requires admin password)
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain boeing_root.pem
```

## Long-Term Maintenance

### Option A: Manual (Simple)

```bash
# Every 3 months, test the bundle
openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout -text | grep "Not After"

# If any cert expires soon (< 30 days), re-download and rebuild bundle
```

### Option B: Automated (Recommended)

Use the included update-boeing-certs.sh script:

```bash
# Copy it
cp update-boeing-certs.sh ~/bin/
chmod +x ~/bin/update-boeing-certs.sh

# Run manually (checks expiration, tests TLS)
~/bin/update-boeing-certs.sh --check-only

# Or auto-update if issues found
~/bin/update-boeing-certs.sh
```

### Option C: Fully Automated (macOS)

Set up weekly launchd check:

```bash
# Copy plist
cp local.boeing-ca-update.plist ~/Library/LaunchAgents/

# Load it
launchctl load ~/Library/LaunchAgents/local.boeing-ca-update.plist

# Verify
launchctl list | grep boeing

# Check logs
tail -f /var/log/boeing-ca-update.log
```

## Migrating from --trusted-host to Proper TLS

### Step 1: Remove the workaround

```bash
# Find where --trusted-host is (if anywhere)
grep -r "trusted-host" ~/.config/pip/ ~/.condarc 2>/dev/null

# Remove it (it shouldn't be needed after you set up the bundle)
```

### Step 2: Test pip without workaround

```bash
source ~/.zshenv
pip3 install poetry
```

If it fails with "certificate verify failed," see Troubleshooting below.

### Step 3: Migrate conda environments

```bash
# Create a test environment and install from Artifactory
conda create -n test-env python=3.13
conda activate test-env

# If this works, your conda config is correct
conda install poetry
```

## Multiple OpenSSL Versions

If you have both system and Homebrew openssl, clarify which one to use:

```bash
# Check what you have
which openssl                    # System
brew list --formula | grep openssl  # Homebrew
ls /usr/local/opt/openssl*/bin   # Intel Homebrew
ls /opt/homebrew/opt/openssl*/bin  # Apple Silicon Homebrew

# Set preference in ~/.zshenv (choose one)
export PATH="/usr/bin:$PATH"                           # Use system (default)
export PATH="/usr/local/opt/openssl@3/bin:$PATH"      # Use Homebrew Intel
export PATH="/opt/homebrew/opt/openssl@3/bin:$PATH"   # Use Homebrew Apple Silicon
```

For the install/update scripts, you can force a specific openssl:

```bash
OPENSSL_BIN="/usr/local/opt/openssl@3/bin/openssl" ~/bin/update-boeing-certs.sh
```

## Troubleshooting

### pip still fails: "certificate verify failed"

1. **Check SSL_CERT_FILE is set:**
   ```bash
   source ~/.zshenv
   echo $SSL_CERT_FILE
   ls -la $SSL_CERT_FILE
   ```

2. **Check bundle has valid certs:**
   ```bash
   openssl crl2pkcs7 -nocrl -certfile ~/.ssl/certs/boeing-ca-bundle.pem | openssl pkcs7 -print_certs -noout > /dev/null && echo OK || echo FAIL
   ```

3. **Test with explicit cert:**
   ```bash
   pip3 install --cert ~/.ssl/certs/boeing-ca-bundle.pem poetry
   ```

4. **Check if Artifactory cert is in bundle:**
   ```bash
   openssl s_client -showcerts -connect sres.web.boeing.com:443 </dev/null 2>&1 | grep "subject=" | head -5
   # Compare CN against your bundle contents
   ```

5. **If cert is missing, add it:**
   ```bash
   # Download just that issuing CA and append to bundle
   curl -L "https://crl.boeing.com/crl/Boeing%20BAS%20Issuing%20CA%20SHA256%20G4.crt" >> ~/.ssl/certs/boeing-ca-bundle.pem
   ```

### conda still fails

1. **Verify .condarc:**
   ```bash
   cat ~/.condarc
   ```
   Should show: `ssl_verify: /Users/so714f/.ssl/certs/boeing-ca-bundle.pem`

2. **Test conda channel:**
   ```bash
   source ~/.zshenv
   conda search numpy
   ```

3. **If still failing, try explicit cert:**
   ```bash
   conda install --ssl-verify ~/.ssl/certs/boeing-ca-bundle.pem numpy
   ```

### Different openssl being used than expected

```bash
which openssl
openssl version

# If wrong version, adjust PATH (see "Multiple OpenSSL Versions" above)
echo $PATH
```

## Files Created / Modified

```
~/.ssl/certs/
  ├─ boeing-ca-bundle.pem           (canonical bundle)
  ├─ boeing-ca-bundle.pem.old-*    (backups)
  ├─ Boeing_BAS_*.crt              (individual certs)
  └─ backups/

~/.config/pip/
  └─ pip.conf                      (cert path for pip)

~/.zshenv                              (SSL env vars - sourced by all shells)
~/.condarc                             (conda ssl_verify setting)
```

## Next Steps

1. **Immediate:** Rebuild your bundle using the corrected commands above
2. **Test:** Run the verification commands to confirm pip/conda/curl work
3. **Automate:** Set up the update script or launchd job for ongoing maintenance
4. **Document:** Keep a note of which Boeing CAs are in your bundle and when they were added

## Questions?

Refer to:
- README.md (full documentation)
- install-boeing-certs.sh (interactive setup with all options)
- update-boeing-certs.sh (periodic verification & auto-update)
- https://boeing.sharepoint.us/sites/IAMPKI (official Boeing PKI info)
