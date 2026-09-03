#!/usr/bin/env bash
# One-time TLS setup for the latent tln<->ucn link.
# Run ON TLN. Creates a local CA + server cert for ucn under
# ~/experiments/tls/ (NOT in git), deploys key+cert to ucn.
#
# Trust model note: this encrypts/authenticates the transport against a
# network bystander and spoofed endpoints on the LAN.  It gives NO privacy
# credit against the declared threat model (UCN itself compromised): the
# TLS private key lives on UCN, which the adversary owns.  Privacy still
# rests entirely on the pre-send obfuscation.
set -euo pipefail

TLS_DIR="$HOME/experiments/tls"
mkdir -p "$TLS_DIR"
cd "$TLS_DIR"

if [[ ! -f ca.key ]]; then
  openssl genrsa -out ca.key 4096
  openssl req -x509 -new -nodes -key ca.key -sha256 -days 825 \
    -subj "/CN=dtraining-latent-link-ca" -out ca.crt
fi

cat > ucn.ext <<'EOF'
subjectAltName=DNS:ucn,IP:10.10.10.2
extendedKeyUsage=serverAuth
EOF

openssl genrsa -out ucn-server.key 2048
openssl req -new -key ucn-server.key -subj "/CN=ucn" -out ucn-server.csr
openssl x509 -req -in ucn-server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -days 825 -sha256 -extfile ucn.ext -out ucn-server.crt
rm -f ucn-server.csr ucn.ext

ssh ucn 'mkdir -p ~/experiments/tls'
scp -q ucn-server.key ucn-server.crt ucn:experiments/tls/
ssh ucn 'chmod 600 ~/experiments/tls/ucn-server.key'
chmod 600 ca.key
echo "TLS material ready: $TLS_DIR (ca.crt is the pinned client anchor)"
