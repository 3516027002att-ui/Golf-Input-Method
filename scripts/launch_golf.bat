@echo off
REM ============================================
REM  golf 系统输入法 — 桌面双击启动器
REM  启动后在任意应用中使用中文输入
REM ============================================
REM  快捷键:
REM    Ctrl+Shift  切换中/英文模式
REM    关闭控制台窗口即可退出
REM ============================================

cd /d "%~dp0.."

REM 系统 IME 模式 (全局可用)
python -m src.input_method.ime_app

REM 如需使用自带编辑器 Demo，改为:
REM python -m src.input_method.app

pause
