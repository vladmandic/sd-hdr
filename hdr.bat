@echo off
setlocal

set PYTHON=python
set VENV=venv

echo HDR: launch
%PYTHON% -c "import venv" 2>nul
if errorlevel 1 (
    echo HDR error: python or venv not installed
    exit /b 1
)

if not exist %VENV% (
    echo HDR: create
    %PYTHON% -m venv %VENV%
    set INITIAL=1
)

if exist %VENV%\Scripts\activate.bat (
    echo HDR: activate
    call %VENV%\Scripts\activate.bat
) else (
    echo HDR error: venv cannot activate
    exit /b 1
)

if defined INITIAL (
    echo HDR: install
    %PYTHON% -m pip install -r requirements.txt
)

echo HDR: exec
%PYTHON% hdr.py %*