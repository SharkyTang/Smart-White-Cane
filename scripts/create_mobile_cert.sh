#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
certificate_dir="$project_dir/.mobile-certs"
if [[ -n "${OPENSSL_BIN:-}" ]]; then
  openssl_bin="$OPENSSL_BIN"
elif [[ -x "/opt/homebrew/bin/openssl" ]]; then
  openssl_bin="/opt/homebrew/bin/openssl"
else
  openssl_bin="$(command -v openssl)"
fi

detect_lan_ip() {
  local interface_name
  interface_name="$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}')"
  if [[ -n "$interface_name" ]]; then
    local detected
    detected="$(ipconfig getifaddr "$interface_name" 2>/dev/null || true)"
    if [[ -n "$detected" ]]; then
      echo "$detected"
      return
    fi
  fi
  ifconfig 2>/dev/null | awk '
    /^[[:alnum:]]+:/ { interface_name=$1; sub(":", "", interface_name) }
    $1 == "inet" && $2 !~ /^127\./ && interface_name !~ /^(lo|utun|awdl|llw)/ {
      print $2
      exit
    }
  '
}

lan_ip="${1:-$(detect_lan_ip)}"
if [[ -z "$lan_ip" ]]; then
  echo "无法自动识别 Mac 的局域网 IP。" >&2
  echo "请手动运行：scripts/create_mobile_cert.sh 192.168.x.x" >&2
  exit 2
fi
if [[ ! "$lan_ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
  echo "局域网 IP 格式不正确：$lan_ip" >&2
  exit 2
fi

local_host="$(scutil --get LocalHostName 2>/dev/null || hostname -s)"
local_name="${local_host}.local"

mkdir -p "$certificate_dir"
chmod 700 "$certificate_dir"

ca_key="$certificate_dir/mobile-ca.key"
ca_cert="$certificate_dir/mobile-ca.crt"
ca_profile="$certificate_dir/mobile-ca.cer"
server_key="$certificate_dir/mobile-server.key"
server_csr="$certificate_dir/mobile-server.csr"
server_cert="$certificate_dir/mobile-server.crt"

if [[ -e "$ca_key" || -e "$ca_cert" ]]; then
  if [[ ! -f "$ca_key" || ! -f "$ca_cert" ]]; then
    echo "证书目录中只有部分 CA 文件；为避免误覆盖，请先检查：$certificate_dir" >&2
    exit 2
  fi
else
  "$openssl_bin" req \
    -x509 \
    -newkey rsa:2048 \
    -sha256 \
    -nodes \
    -days 3650 \
    -subj "/CN=Smart Cane Local Test CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -keyout "$ca_key" \
    -out "$ca_cert"
fi

extension_file="$(mktemp "${TMPDIR:-/tmp}/smart-cane-cert.XXXXXX")"
trap 'rm -f "$extension_file"' EXIT
{
  echo "basicConstraints=critical,CA:FALSE"
  echo "keyUsage=critical,digitalSignature,keyEncipherment"
  echo "extendedKeyUsage=serverAuth"
  echo "subjectAltName=IP:$lan_ip,IP:127.0.0.1,DNS:$local_name,DNS:localhost"
} > "$extension_file"

"$openssl_bin" req \
  -new \
  -newkey rsa:2048 \
  -sha256 \
  -nodes \
  -subj "/CN=$lan_ip" \
  -keyout "$server_key" \
  -out "$server_csr"

"$openssl_bin" x509 \
  -req \
  -sha256 \
  -days 397 \
  -in "$server_csr" \
  -CA "$ca_cert" \
  -CAkey "$ca_key" \
  -CAcreateserial \
  -extfile "$extension_file" \
  -out "$server_cert"

"$openssl_bin" x509 -in "$ca_cert" -outform der -out "$ca_profile"
chmod 600 "$ca_key" "$server_key"
rm -f "$server_csr" "$certificate_dir/mobile-ca.srl"

echo
echo "手机测试证书已生成：$certificate_dir"
echo "iPhone 安装文件：$ca_profile"
echo "手机访问地址：https://$lan_ip:8443"
echo
echo "下一步："
echo "1. 通过隔空投送把 mobile-ca.cer 发到 iPhone 并安装描述文件。"
echo "2. 在 iPhone 的 设置 → 通用 → 关于本机 → 证书信任设置 中启用完全信任。"
echo "3. 运行 uv run smart-cane-mobile，再用 Safari 打开上面的地址。"
