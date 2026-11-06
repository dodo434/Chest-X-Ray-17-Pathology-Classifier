#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f ".qt-libs/extracted/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0.0.0" ]]; then
    command -v apt-get >/dev/null || {
        echo "apt-get is required to download the Qt X11 compatibility library." >&2
        exit 1
    }
    mkdir -p .qt-libs
    (cd .qt-libs && apt-get download libxcb-cursor0)
    dpkg-deb -x .qt-libs/libxcb-cursor0_*.deb .qt-libs/extracted
fi

source .venv/bin/activate
export LD_LIBRARY_PATH="$PWD/.qt-libs/extracted/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python3 main.py
