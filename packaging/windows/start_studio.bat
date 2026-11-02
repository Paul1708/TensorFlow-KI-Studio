@echo off
chcp 65001 >nul
setlocal
set "REPO_ROOT=%~dp0..\.."
cd /d "%REPO_ROOT%"
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUTF8=1
set "PYTHONPATH=%REPO_ROOT%\src"

python -c "import sys" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    where pythonw >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        start "" pythonw -m ki_studio
        exit /b 0
    )
    start "" python -m ki_studio
    exit /b 0
)

where pyw >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start "" pyw -3 -m ki_studio
    exit /b 0
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start "" py -3 -m ki_studio
    exit /b 0
)

echo [Hinweis] Python wurde auf diesem PC nicht gefunden.
echo Bitte Python 3.10+ installieren und einmalig ausfuehren:
echo     pip install -r requirements.txt
echo.
pause
