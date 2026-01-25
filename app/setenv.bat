:: Creates a directory under your userprofile for all virtual enviroments
:: And then creates you enviroment under it. The enviroment will have the same name
:: as the parent directory this batch file is in (assumed to be the project name)
::
:: This avoids large directory with binaries in my project foldername
::
:: It can also install a numpy/scipy package from the Intel repository.
:: These packages are beter performing and more stable compared to the version from pypy (based on openblas)
:: 
echo off

SET DATA_DIR=D:
:: python base directory
title 'Python 3.13'
SET PYTHON_DIR=%DATA_DIR%\python313
PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%;
:: change to directory this batchfile is in.
cd %~p0% 


::Get parent foldername from the current batfile path.
SET "CDIR=%~dp0"
:: for loop requires removing trailing backslash from %~dp0 output
SET "CDIR=%CDIR:~0,-1%"
FOR %%i IN ("%CDIR%") DO SET "VENV_DIR=%%~nxi"
:: now that we know the parent folder name in VENV_DIR
SET VENV_DIR=%DATA_DIR%\venv\%VENV_DIR%
echo %VENV_DIR%

:: Create enviroment if not existing and activate
:: The enviroment is created in %DATA_DIR%\venv
if exist %VENV_DIR% goto ready
	mkdir %VENV_DIR%

	python -m venv %VENV_DIR%

	call %VENV_DIR%\scripts\activate
	:: remove this if you do not want numpy or scipy
	:: _distributor_init_local.py is also committed in numlib
	:: copy it to the venv rootdir to make this line work
	pip install -i https://software.repos.intel.com/python/pypi numpy scipy
	copy %DATA_DIR%\venv\_distributor_init_local.py %VENV_DIR%\Lib\site-packages\numpy
	:: Generate a constraints file to force pip to use the intel numpy version
	pip freeze > constraints.txt
	pip install -r requirements.txt -c constraints.txt
:ready

call %VENV_DIR%\scripts\activate

cmd
