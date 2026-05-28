#!/usr/bin/env bash
# Install the Linux benchmark tools benchpilot shells out to.
# Arch-family one-liner. Edit if you're on Debian/Fedora.
set -euo pipefail

PKGS=(
    sysbench     # CPU + memory benchmark
    fio          # storage benchmark
    mbw          # memory bandwidth
    stress-ng    # sustained stress for thermal tests
    lm_sensors   # CPU/board temps + fans - Check
    smartmontools # NVMe/SSD telemetry - Check
    nvme-cli     # NVMe queries
)

if command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed --noconfirm "${PKGS[@]}"
elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y sysbench fio mbw stress-ng lm-sensors smartmontools nvme-cli
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y sysbench fio mbw stress-ng lm_sensors smartmontools nvme-cli
else
    echo "Unsupported package manager. Install manually: ${PKGS[*]}" >&2
    exit 1
fi

echo
echo "OK. Next:  uv sync --extra gpu"
echo "Then:     uv run benchpilot run --quick"
