@echo off
REM ----------------------------------------------------------------
REM  TomatoGuard - Windows shortcut to launch the website.
REM  Double-click this file to start.
REM ----------------------------------------------------------------

cd /d "%~dp0"

if not exist "tomato_model.h5" (
    echo.
    echo  [!] The trained model is missing.
    echo  [!] Run the first two steps before using this shortcut:
    echo.
    echo        python 1_prepare_dataset.py
    echo        python 2_train_model.py
    echo.
    pause
    exit /b 1
)

echo Starting TomatoGuard server at http://localhost:8000 ...
python 3_server.py
pause
