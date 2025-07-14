@echo off
REM -*- coding: utf-8 -*-
REM file: upload_packages.bat
REM Upload both omnigenome and omnigenbench packages to PyPI

echo 🚀 Starting package upload process...

REM Function to upload a package
call :upload_package "setup_omnigenome.py" "omnigenome"
call :upload_package "setup_omnigenbench.py" "omnigenbench"

echo 🎉 All packages uploaded successfully!
goto :eof

:upload_package
set setup_file=%~1
set package_name=%~2

echo 📦 Building and uploading %package_name%...

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"

REM Build the package
python %setup_file% sdist bdist_wheel

REM Upload to PyPI (use --repository testpypi for testing)
python -m twine upload dist/*

REM Clean up
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"

echo ✅ %package_name% uploaded successfully!
goto :eof
