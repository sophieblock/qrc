#!/usr/bin/env bash
set -euo pipefail

# Boeing CA Bundle Update & Verification Script
# Purpose: Periodically update and verify Boeing CA bundle (similar to update-ca-certificates on Linux)
# Usage: ./update-boeing-certs.sh [--check-only] [--force]
# Can be run manually or via launchd/cron

BUNDLE_FILE="${HOME}/.ssl/certs/boeing-ca-bundle.pem"
BUNDLE_BACKUP="${HOME}/.ssl/certs/backups/pre-update-$(date -u +%Y%m%dT%H%M%SZ).pem"
OPENSSL_BIN="${OPENSSL_BIN:-openssl}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

CHECK_ONLY=0
FORCE_UPDATE=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --check-only) CHECK_ONLY=1; shift ;;
    --force) FORCE_UPDATE=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

log_info "Boeing CA Bundle Updater"

if [ ! -f "${BUNDLE_FILE}" ]; then
  log_error "Bundle not found at ${BUNDLE_FILE}"
  log_info "Run install-boeing-certs.sh first."
  exit 1
fi

# Check expiration dates
log_info "Checking certificate expiration dates..."

local all_valid=true
while IFS= read -r line; do
  if [[ "${line}" == "Not After"* ]]; then
    expiry_date=$(echo "${line}" | sed 's/.*: //')
    expiry_epoch=$(date -jf "%b %d %T %Y %Z" "${expiry_date}" +%s 2>/dev/null || echo "0")
    current_epoch=$(date +%s)
    days_left=$(( (expiry_epoch - current_epoch) / 86400 ))
    
    if [ "${days_left}" -lt 0 ]; then
      log_error "Certificate expired ${days_left} days ago"
      all_valid=false
    elif [ "${days_left}" -lt 30 ]; then
      log_warn "Certificate expires in ${days_left} days"
    else
      log_info "Certificate valid for ${days_left} more days"
    fi
  fi
done < <("${OPENSSL_BIN}" crl2pkcs7 -nocrl -certfile "${BUNDLE_FILE}" 2>/dev/null | "${OPENSSL_BIN}" pkcs7 -print_certs -noout -text 2>/dev/null | grep "Not After")

if [ "${all_valid}" != "true" ]; then
  log_error "One or more certificates are expired or expiring soon"
  if [ "${FORCE_UPDATE}" != "1" ]; then
    log_info "Run with --force to attempt renewal"
    exit 1
  fi
fi

# Test TLS connections
log_info "Testing TLS connections..."

local test_passed=true
for host in sres.web.boeing.com git.web.boeing.com; do
  if echo "" | timeout 5 "${OPENSSL_BIN}" s_client -CAfile "${BUNDLE_FILE}" -connect "${host}:443" 2>&1 | grep -q "Verify return code: 0 (ok)"; then
    log_info "  ✓ ${host} ok"
  else
    log_warn "  ✗ ${host} failed"
    test_passed=false
  fi
done

if [ "${CHECK_ONLY}" == "1" ]; then
  if [ "${test_passed}" == "true" ] && [ "${all_valid}" == "true" ]; then
    log_info "Bundle is valid and all tests passed"
    exit 0
  else
    log_warn "Bundle has issues; consider running update-boeing-certs.sh (without --check-only) to refresh"
    exit 1
  fi
fi

# Perform update
if [ "${FORCE_UPDATE}" == "1" ] || [ "${test_passed}" != "true" ]; then
  log_info "Backing up current bundle..."
  cp "${BUNDLE_FILE}" "${BUNDLE_BACKUP}"
  log_info "Backup: ${BUNDLE_BACKUP}"
  
  log_info "Re-running installation to refresh certificates..."
  if command -v install-boeing-certs.sh &>/dev/null; then
    NO_PROMPT=1 install-boeing-certs.sh
  elif [ -f "${HOME}/bin/install-boeing-certs.sh" ]; then
    NO_PROMPT=1 "${HOME}/bin/install-boeing-certs.sh"
  else
    log_error "install-boeing-certs.sh not found in PATH or ~/bin"
    log_info "Restore from backup: cp ${BUNDLE_BACKUP} ${BUNDLE_FILE}"
    exit 1
  fi
  
  log_info "Bundle updated successfully"
else
  log_info "Bundle is current; no update needed"
fi

exit 0
