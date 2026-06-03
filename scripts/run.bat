@echo off
REM ============ scripts/run.bat ============
REM 本地开发快捷脚本（Windows CMD 版）
REM 用法: scripts\run.bat [stage]
REM
REM Examples:
REM   scripts\run.bat all       :: 完整训练管线
REM   scripts\run.bat teacher   :: 仅训练教师
REM   scripts\run.bat check     :: 检查环境
REM   scripts\run.bat export    :: 仅导出 + 量化

setlocal

cd /d "%~dp0\.."

echo SkyEye — Weather Classification Pipeline
echo =========================================
echo Stage: %1
echo.

REM 激活虚拟环境（如果存在）
if exist ".venv\Scripts\python.exe" (
    echo Using .venv virtual environment
    set PYTHON=.venv\Scripts\python.exe
) else (
    echo Using system Python
    set PYTHON=python
)

if "%1"=="" (
    set STAGE=all
) else (
    set STAGE=%1
)

%PYTHON% scripts\local_train.py %STAGE%

endlocal
