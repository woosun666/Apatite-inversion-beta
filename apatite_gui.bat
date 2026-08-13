@echo off
rem Double-click launcher for the Ok Tedi apatite GUI.
rem Optional: drag a .otproj file onto this launcher (or associate .otproj
rem with it) to open that project directly.
cd /d "%~dp0"
where pythonw.exe >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw.exe apatite_gui.py %1
) else (
  start "" python.exe apatite_gui.py %1
)
