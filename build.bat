@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo MSVC init failed
    exit /b 1
)

if not exist "build" mkdir build
if not exist "logs" mkdir logs

echo Building geneslime.exe with RDC enabled...
nvcc -arch=sm_86 -std=c++17 -rdc=true -O3 --use_fast_math --expt-relaxed-constexpr -Xptxas -O3 --keep --keep-dir build main.cu -lcudadevrt -o geneslime.exe > logs\build.log 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo Build failed - check logs\build.log
    type logs\build.log
    exit /b %ERRORLEVEL%
)

echo Build complete: geneslime.exe
exit /b 0
