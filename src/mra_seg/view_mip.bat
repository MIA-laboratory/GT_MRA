@echo off
setlocal
REM ============================================================
REM MRA MIP (Maximum Intensity Projection) Comparison Viewer
REM Shows the MIP of each held-out test case:
REM   Original | GT Masked | Initial | Round1 | Round2
REM
REM Controls:
REM   Left/Right arrows or mouse wheel: switch cases
REM   Prev/Next buttons: navigate
REM ============================================================

echo Starting MIP Comparison Viewer...
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
"%PYTHON%" viewer_mip.py

if errorlevel 1 (
    echo.
    echo ERROR: Viewer failed to start.
    pause
)

:END
endlocal
