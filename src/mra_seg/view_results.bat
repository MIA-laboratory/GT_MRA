@echo off
setlocal
REM ============================================================
REM MRA Segmentation Overlay Viewer
REM Compares the initial model with an evolved model
REM on the held-out test cases
REM
REM Controls:
REM   Mouse wheel : scroll through slices
REM   Arrow keys  : navigate slices
REM   Home / End  : jump to first / last slice
REM   Slider      : direct navigation
REM
REM Red overlay   = model prediction
REM Green contour = ground truth
REM ============================================================

echo Starting MRA Segmentation Overlay Viewer...
echo.

REM Use the bundled embedded Python via a relative path (portable).
set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%..\..\python\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Embedded Python not found at "%PYTHON%"
    pause
    goto :END
)

cd /d "%SCRIPT_DIR%"
"%PYTHON%" viewer_overlay.py

if errorlevel 1 (
    echo.
    echo ERROR: Viewer failed to start.
    echo Make sure Python and required packages are installed.
    pause
)

:END
endlocal
