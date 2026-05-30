@echo off
REM golf 输入法桌面启动器
REM 双击此文件即可启动 golf（无需命令行）

cd /d "%~dp0.."
python -m src.input_method.tray_app
pause
