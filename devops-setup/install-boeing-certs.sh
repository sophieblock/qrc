#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Boeing CA Bundle Installation & Verification Script for macOS/Linux
# Purpose: Download Boeing CA certificates, build a canonical PEM bundle,
# verify chain, and configure Python/pip/conda/curl to use it.
# Usage: ./install-boeing-certs.sh [--no-prompt] [--openssl-bin /path/to/openssl]

# CONFIG
BUNDLE_DIR="${HOME}/.ssl/certs"
BUNDLE_FILE="${BUNDLE_DIR}/boeing-ca-bundle.pem"
BACKUP_DIR="${HOME}/.ssl/certs/backups/$(date -u +%Y%m%dT%H%M%SZ)"
PIP_CONFIG_DIR="${HOME}/.config/pip"
PIP_CONFIG="${PIP_CONFIG_DIR}/pip.conf"
CONDA_RC="${HOME}/.condarc"
ZSHENV="${HOME}/.zshenv"
ZSHRC="${HOME}/.zshrc"

# Allow overriding openssl binary
OPENSSL_BIN="${OPENSSL_BIN:-openssl}"

# Boeing cert URLs (from Boeing CRL server, no %20 encoding)
declare -a BOEING_URLS=(
  "https://crl.boeing.com/crl/Boeing%20BAS%20Issuing%20CA%20SHA256%20G4.crt"
  "https://crl.boeing.com/crl/Boeing%20BAS%20Root%20CA%20SHA256%20G3.crt"
  "https://crl.boeing.com/crl/Boeing%20Basic%20Assurance%20Software%20Issuing%20CA%20G3.crt"
  "https://crl.boeing.com/crl/Boeing%20Basic%20Assurance%20Software%20Root%20CA%20G2.crt"
)

# Test hosts
TEST_HOSTS=("sres.web.boeing.com" "git.web.boeing.com")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper: log with color
log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_section() { echo -e "\n${GREEN}=== $* ===${NC}\n"; }

# Helper: prompt user
prompt_yes_no() {
  local msg="$1"
  local default="${2:-y}"
  if [[ "${NO_PROMPT:-}" == "1" ]]; then
    return 0  # assume yes
  fi
  read -p "${msg} [${default}]: " -r response
  [[ -z "$response" ]] && response="$default"
  [[ "$response" =~ ^[Yy]$ ]]
}

# Detect openssl location
if command -v "${OPENSSL_BIN}" &>/dev/null; then
  log_info "Using openssl: $(command -v "${OPENSSL_BIN}")"
  OPENSSL_BIN="$(command -v "${OPENSSL_BIN}")"
else
  log_error "OpenSSL not found at ${OPENSSL_BIN}. Install openssl or set OPENSSL_BIN environment variable."
  exit 1
fi

log_section "Boeing CA Bundle Installer"

# Create directories
log_info "Creating certificate directories..."
mkdir -p "${BUNDLE_DIR}" "${BACKUP_DIR}" "${PIP_CONFIG_DIR}"

# Backup existing bundle
if [ -f "${BUNDLE_FILE}" ]; then
  log_warn "Existing bundle found. Backing up to ${BACKUP_DIR}"
  cp -a "${BUNDLE_FILE}" "${BACKUP_DIR}/boeing-ca-bundle.pem.bak"
  log_info "Backup saved."
fi

# Create fresh bundle
log_info "Creating fresh bundle file: ${BUNDLE_FILE}"
> "${BUNDLE_FILE}"  # truncate
chmod 644 "${BUNDLE_FILE}"

# Function to download and append cert
download_and_append() {
  local url="$1"
  local tmpfile tmppem
  
  tmpfile="$(mktemp)" || { log_error "mktemp failed"; return 1; }
  tmppem="$(mktemp)" || { log_error "mktemp failed"; return 1; }
  trap "rm -f '${tmpfile}' '${tmppem}'" RETURN

  log_info "Downloading: ${url}"
  if ! curl --retry 5 --fail --silent -L -o "${tmpfile}" "${url}"; then
    log_error "Failed to download ${url}"
    return 1
  fi

  # Detect if DER or PEM and convert to PEM
  if file "${tmpfile}" | grep -q "data" || ! grep -q "BEGIN CERTIFICATE" "${tmpfile}" 2>/dev/null; then
    log_info "  Converting DER → PEM..."
    if ! "${OPENSSL_BIN}" x509 -inform DER -in "${tmpfile}" -out "${tmppem}"; then
      log_error "DER conversion failed for ${url}"
      return 1
    fi
  else
    # Already PEM
    cp "${tmpfile}" "${tmppem}"
  fi

  # Verify certificate validity
  if ! "${OPENSSL_BIN}" x509 -noout -in "${tmppem}" &>/dev/null; then
    log_error "Certificate validation failed for ${url}"
    return 1
  fi

  # Show certificate info
  local subject issuer notafter
  subject=$("${OPENSSL_BIN}" x509 -noout -subject -in "${tmppem}" | sed 's/subject=//')
  issuer=$("${OPENSSL_BIN}" x509 -noout -issuer -in "${tmppem}" | sed 's/issuer=//')
  notafter=$("${OPENSSL_BIN}" x509 -noout -enddate -in "${tmppem}" | sed 's/notAfter=//')
  
  log_info "  Subject: ${subject}"
  log_info "  Issuer: ${issuer}"
  log_info "  Expires: ${notafter}"

  # Append to bundle
  cat "${tmppem}" >> "${BUNDLE_FILE}"
  log_info "  ✓ Added to bundle"
  return 0
}

log_section "Step 1: Download and Build Bundle"

for url in "${BOEING_URLS[@]}"; do
  download_and_append "${url}" || log_warn "Skipping ${url} due to error"
done

log_section "Step 2: Verify Bundle Contents"

log_info "Bundle file: ${BUNDLE_FILE}"
log_info "Bundle size: $(wc -c < "${BUNDLE_FILE}") bytes"

local cert_count
cert_count=$(grep -c "BEGIN CERTIFICATE" "${BUNDLE_FILE}")
log_info "Certificate count: ${cert_count}"

if [ "${cert_count}" -eq 0 ]; then
  log_error "No certificates found in bundle. Exiting."
  exit 1
fi

echo ""
log_info "Bundle contents:"
"${OPENSSL_BIN}" crl2pkcs7 -nocrl -certfile "${BUNDLE_FILE}" | "${OPENSSL_BIN}" pkcs7 -print_certs -noout -text | grep -E "Subject:|Issuer:|Not After" | head -20

log_section "Step 3: Test TLS Connections"

for host in "${TEST_HOSTS[@]}"; do
  log_info "Testing ${host}..."
  if echo "" | timeout 10 "${OPENSSL_BIN}" s_client -CAfile "${BUNDLE_FILE}" -connect "${host}:443" 2>&1 | grep -q "Verify return code: 0 (ok)"; then
    log_info "  ✓ ${host} verification successful"
  else
    log_warn "  ✗ ${host} verification failed (may be network-related)"
  fi
done

log_section "Step 4: Configure Python/pip/conda"

# Add SSL environment variables to ~/.zshenv
log_info "Configuring ~/.zshenv..."
if grep -q "SSL_CERT_FILE" "${ZSHENV}" 2>/dev/null; then
  log_info "  SSL_CERT_FILE already in ~/.zshenv, skipping"
else
  cat >> "${ZSHENV}" <<EOF

# Boeing CA Bundle (added by install-boeing-certs.sh)
export SSL_CERT_FILE="\${HOME}/.ssl/certs/boeing-ca-bundle.pem"
export REQUESTS_CA_BUNDLE="\${HOME}/.ssl/certs/boeing-ca-bundle.pem"
EOF
  log_info "  ✓ Added SSL_CERT_FILE and REQUESTS_CA_BUNDLE to ~/.zshenv"
fi

# Configure pip
log_info "Configuring pip..."
if [ ! -f "${PIP_CONFIG}" ]; then
  mkdir -p "${PIP_CONFIG_DIR}"
  cat > "${PIP_CONFIG}" <<EOF
[global]
cert = ${HOME}/.ssl/certs/boeing-ca-bundle.pem
EOF
  log_info "  ✓ Created ${PIP_CONFIG}"
else
  log_warn "  pip.conf already exists. Review manually: ${PIP_CONFIG}"
fi

# Configure conda
log_info "Configuring conda..."
if [ ! -f "${CONDA_RC}" ]; then
  cat > "${CONDA_RC}" <<EOF
ssl_verify: ${HOME}/.ssl/certs/boeing-ca-bundle.pem
EOF
  log_info "  ✓ Created ${CONDA_RC}"
elif ! grep -q "ssl_verify" "${CONDA_RC}" 2>/dev/null; then
  cat >> "${CONDA_RC}" <<EOF
ssl_verify: ${HOME}/.ssl/certs/boeing-ca-bundle.pem
EOF
  log_info "  ✓ Added ssl_verify to ${CONDA_RC}"
else
  log_warn "  ssl_verify already in ${CONDA_RC}. Review manually."
fi

log_section "Step 5: Optional – Append to System certifi"

if prompt_yes_no "Append Boeing certs to Python certifi bundle? (backup created)"; then
  local certifi_path
  certifi_path=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null) || {
    log_warn "Python certifi not available, skipping"
    certifi_path=""
  }
  
  if [ -n "${certifi_path}" ] && [ -f "${certifi_path}" ]; then
    log_info "Certifi path: ${certifi_path}"
    mkdir -p "${CERTIFI_SAFE_BACKUP_DIR}"
    cp "${certifi_path}" "${CERTIFI_SAFE_BACKUP_DIR}/cacert.pem.bak"
    log_info "  Backup: ${CERTIFI_SAFE_BACKUP_DIR}/cacert.pem.bak"
    
    # Append only if not already present
    if grep -q "Boeing" "${certifi_path}" 2>/dev/null; then
      log_info "  Boeing certs already in certifi, skipping"
    else
      cat "${BUNDLE_FILE}" >> "${certifi_path}"
      log_info "  ✓ Appended Boeing bundle to certifi"
    fi
  fi
fi

log_section "Step 6: Optional – Import Root to macOS Keychain"

if [[ "${OSTYPE}" == "darwin"* ]]; then
  if prompt_yes_no "Import Boeing root CA to macOS System Keychain? (requires sudo)"; then
    # Extract first cert (issuing CA) and last cert (root CA)
    local root_pem
    root_pem="$(mktemp)"
    trap "rm -f '${root_pem}'" RETURN
    
    # Get the last certificate (root)
    awk '/-----BEGIN CERTIFICATE-----/{i++; flag=1} flag{print > "'"${root_pem}"'"." i} /-----END CERTIFICATE-----/{flag=0}' "${BUNDLE_FILE}"
    
    local last_cert
    last_cert=$(ls "${root_pem}".* 2>/dev/null | tail -1) || { log_warn "Could not extract root cert"; return 0; }
    
    log_info "Root cert: ${last_cert}"
    
    # Import to System Keychain
    sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "${last_cert}" && {
      log_info "  ✓ Root CA imported to System Keychain"
    } || {
      log_warn "  Failed to import to System Keychain (may require interactive approval)"
    }
    
    rm -f "${root_pem}".* 2>/dev/null || true
  fi
fi

log_section "Step 7: Verification Commands"

cat <<EOF

To verify everything is working, run these commands:

1. Source the new environment:
   source ~/.zshenv

2. Verify Python/pip use the bundle:
   python3 -c "import ssl,os; print('SSL_CERT_FILE:', os.environ.get('SSL_CERT_FILE')); print(ssl.get_default_verify_paths())"

3. Test curl:
   curl -v https://sres.web.boeing.com/

4. Test pip:
   pip3 install --dry-run poetry 2>&1 | head -20

5. Test conda:
   conda search numpy 2>&1 | head -10

6. Check bundle expiration:
   "${OPENSSL_BIN}" crl2pkcs7 -nocrl -certfile "${BUNDLE_FILE}" | "${OPENSSL_BIN}" pkcs7 -print_certs -noout -notext -text | grep "Not After"

EOF

log_section "Installation Complete"

log_info "Bundle location: ${BUNDLE_FILE}"
log_info "Backups: ${BACKUP_DIR}"
log_info "Configuration files:"
log_info "  - ${ZSHENV}"
log_info "  - ${PIP_CONFIG}"
log_info "  - ${CONDA_RC}"
log_info ""
log_info "Next: run 'source ~/.zshenv' and verify with commands above."
log_info ""
log_info "To update in the future: re-run this script or run update-boeing-certs.sh"

exit 0
