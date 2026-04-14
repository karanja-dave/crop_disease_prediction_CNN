@echo off
echo Setting up NeuralNest Environment...

cd /d "D:\CAPSTONE REVISED"

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo Creating folders...
mkdir models 2>nul
mkdir logs 2>nul
mkdir notebooks 2>nul
mkdir src 2>nul

echo Setup complete!
echo.
echo Next steps:
echo 1. Open VS Code in this folder
echo 2. Select Python interpreter: ./venv/Scripts/python.exe
echo 3. Run data_preparation.ipynb
pause