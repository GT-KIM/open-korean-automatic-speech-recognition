#!/usr/bin/env bash
# Install Docker and GPU container support; never change GPU drivers or reboot.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then echo 'Run with sudo.' >&2; exit 1; fi
. /etc/os-release
if [ "$ID" != ubuntu ] || ! [[ "$VERSION_ID" =~ ^(22\.04|24\.04)$ ]]; then
  echo 'This installer supports Ubuntu 22.04/24.04 only.' >&2; exit 1
fi
if command -v docker >/dev/null && [ -n "$(docker ps -q)" ]; then
  echo 'Existing containers are running; inspect them before configuring Docker.' >&2; exit 1
fi
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l
apt-get update -qq
apt-get install -y --no-install-recommends ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
packages=(docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit)
plan=$(apt-get -s --no-install-recommends install "${packages[@]}")
if printf '%s\n' "$plan" | grep -Eq '^Remv |^Inst (nvidia-driver|nvidia-dkms|nvidia-kernel|libnvidia-(compute|gl|encode|decode)|linux-image|linux-headers)'; then
  printf '%s\n' "$plan"
  echo 'Unexpected driver/kernel/removal changes; stop for review.' >&2; exit 1
fi
apt-get install -y --no-install-recommends "${packages[@]}"
systemctl enable --now docker
if [ -n "$(docker ps -q)" ]; then
  echo 'Containers appeared during installation; do not restart Docker.' >&2; exit 1
fi
if [ -f /etc/docker/daemon.json ] && [ ! -f /etc/docker/daemon.json.openkoasr-before ]; then
  cp -p /etc/docker/daemon.json /etc/docker/daemon.json.openkoasr-before
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker info --format '{{json .Runtimes}}'
dpkg-query -W docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
