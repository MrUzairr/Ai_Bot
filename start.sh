#!/bin/bash
set -eu

export PYTHONUNBUFFERED=true

VIRTUALENV=".data/venv"

if [ ! -d "$VIRTUALENV" ]; then
    python3 -m venv "$VIRTUALENV"
    source "$VIRTUALENV/bin/activate"
    pip install -r requirements.txt
else
    source "$VIRTUALENV/bin/activate"
fi

if [ ! -f "$VIRTUALENV/bin/pip" ]; then
    curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | "$VIRTUALENV/bin/python"
fi

"$VIRTUALENV/bin/pip" install -r requirements.txt
"$VIRTUALENV/bin/python" main.py
