# Scalable_ML

This project template can be used to work on the programming exercises. 
Note that the following routines have been tested on a Linux machine and might not work on other OS.

## Create a virtual environment or a conda environment
It is common and recommended practice to use virtual environments for work in Python. 
Either a virtual environment or a conda environment can be used for this. A Python3 interpreter is sufficient to set up a virtual environment. 
For conda, for example, the conda-forge Python distribution (similar to Anaconda) can be used. 
In both cases, this has to be done before the pip install statement

## Setup virtual environment with name env-name (has been tested at AIfA CIP pool)
python3 -m venv scalable_ml

source scalable_ml/bin/activate

## Setup a conda environment (alternative to virtual environment, requires conda-forge not directly available at AIfA CIP pool)
conda update conda

conda create -n scalable_ml_env python=3.9

conda activate scalable_ml_env


## Install the project locally
If needed update pip to the newest version, otherwise the installation might fail:

pip install -U pip

In order to install the package do

pip install -e ".[test]"

If you don’t do this “editable installation” (i.e. pip install "."),  then your tests won’t run because the package will not be installed. An editable install means that changes in the code will be immediately reflected in the functionality of the package.
During the installation process, Python setuptools reads the file pyproject.toml and installs all packages that are listed there. If you require further packages, than you should add it to the pyproject.toml files. In the past, an alternative for the configuration of setuptools was done using a file called setup.cfg. It is only necessary to use on of these files, either pyproject.toml (newer) or setup.cfg (older).

## Potentially update your PYTHONPATH variable

It may not be possible to execute the Python code directly after installation because the Python search path has not been adjusted, i.e. the new search path has to be added. There are IDEs, such as PyCharm, that do this automatically, so it is unlikely then that the search path needs to be adjusted. If this is not the case, an adjustment might be necessary.
Please note that "aruettg1" is not your Uni Bonn ID so that you have to adapt the following command for the AIfA CIP pool to your specific needs

export PYTHONPATH="${PYTHONPATH}:/home/aruettg1/physik/scalable_ml/scalable_ml/lib/python3.8/site-packages/scalable_ml/"

This command can be written directly into the file ~/.bashrc and does not then have to be executed every time the terminal is restarted.