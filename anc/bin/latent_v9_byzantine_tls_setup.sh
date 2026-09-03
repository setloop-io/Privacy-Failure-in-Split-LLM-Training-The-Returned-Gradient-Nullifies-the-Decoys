#!/usr/bin/env bash
# Issue per-node server certs for the CPU Byzantine nodes, signed by the
# existing latent-link CA on tln.  Run ON TLN.
set -euo pipefail
TLS_DIR="$HOME/experiments/tls"
cd "$TLS_DIR"
[ -f ca.key ] || { echo "CA missing; run latent_v7_tls_setup.sh first"; exit 1; }

for ip in 192.168.1.120 192.168.1.121 192.168.1.122; do
  name="byz-$ip"
  printf 'subjectAltName=IP:%s\nextendedKeyUsage=serverAuth\n' "$ip" > "$name.ext"
  openssl genrsa -out "$name.key" 2048
  openssl req -new -key "$name.key" -subj "/CN=$ip" -out "$name.csr"
  openssl x509 -req -in "$name.csr" -CA ca.crt -CAkey ca.key -CAcreateserial \
    -days 825 -sha256 -extfile "$name.ext" -out "$name.crt"
  rm -f "$name.csr" "$name.ext"
  scp -q "$name.key" "$name.crt" "geo@$ip:experiments/tls/" 2>/dev/null || {
    ssh geo@$ip 'mkdir -p ~/experiments/tls'
    scp -q "$name.key" "$name.crt" "geo@$ip:experiments/tls/"
  }
  ssh geo@$ip "chmod 600 ~/experiments/tls/$name.key"
  echo "issued $ip"
done
