@echo off
setlocal
REM ============================================================
REM MRA Segmentation Triple Overlay Viewer
REM Compares 3 models side-by-side:
REM   Left:   Initial Model (Gen 0)
REM   Center: Round 1 Gen 10 (Naive self-training)
REM   Right:  Round 2 Final (Improved evolutionary)
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

echo Starting Triple Overlay Viewer...
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
"%PYTHON%" viewer_overlay_v2.py

if errorlevel 1 (
    echo.
    echo ERROR: Viewer failed to start.
    pause
)

:END
endlocal
