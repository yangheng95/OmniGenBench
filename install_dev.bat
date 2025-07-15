@echo off
echo Uninstalling existing packages...
pip uninstall omnigenbench omnigenome -y

echo Installing package in development mode...
pip install -e .

echo Verifying installation...
pip show omnigenbench
echo.
echo Testing entry points...
python -c "import omnigenbench; print('omnigenbench imported successfully')"
echo.
echo Checking console scripts...
where autobench
where autotrain

echo.
echo Installation complete! You can now use:
echo   autobench -m model_name -b benchmark_name
echo   autotrain -m model_name -d dataset_name
pause
