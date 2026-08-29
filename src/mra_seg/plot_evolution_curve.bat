@echo off
setlocal
REM ============================================================
REM Evolution Curve Visualization
REM Plots Round 1 (Naive) vs Round 2 (Improved) learning curves
REM Saves figures under results/mra_seg/figures/
REM ============================================================

echo Generating evolution curve plots...
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
"%PYTHON%" plot_evolution.py

if errorlevel 1 (
    echo.
    echo ERROR: Plot generation failed.
    pause
)

:END
endlocal
