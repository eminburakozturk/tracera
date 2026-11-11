@echo off
REM Tracera - ortam kurulumu (bir kez calistirilir)
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
call conda create -n tracera python=3.11 -y
call conda activate tracera
python -m pip install --upgrade pip
REM RTX 4060 icin CUDA 12.6 destekli PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
echo.
echo ===== ORTAM KONTROLU =====
python scripts\check_env.py
pause
