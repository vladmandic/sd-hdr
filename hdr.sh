#!/usr/bin/env bash

PYTHON="python3"
VENV="venv"

echo "HDR: launch"
if ! "${PYTHON}" -c "import venv" &>/dev/null; then
  echo "HDR error: python or venv not installed"
  exit 1
fi
if [[ ! -d "${VENV}" ]]; then
  echo "HDR: create"
  "${PYTHON}" -m venv "${VENV}"
  INITIAL=1
fi
if [[ -f "${VENV}"/bin/activate ]]; then
  echo "HDR: activate"
  source "${VENV}"/bin/activate
else
  echo "HDR error: venv cannot activate"
  exit 1
fi
if [[ ! -z ${INITIAL+x} ]]; then
  echo "HDR: install"
  "${PYTHON}" -m pip install -r requirements.txt
fi
echo "HDR: exec"
exec "${PYTHON}" hdr.py "$@"
