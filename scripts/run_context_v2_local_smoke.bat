@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_context_v2_local_smoke.ps1" %*
exit /b %ERRORLEVEL%
