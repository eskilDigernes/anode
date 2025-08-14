python -m venv .venv

Activate the Virtual Environment
Then, activate it by running:

.\.venv\Scripts\Activate.ps1


Install Dependencies
With the virtual environment active, install required dependencies:


pip install open3d matplotlib numpy

Your .venv folder is already excluded in your .gitignore. After these steps, you can run your script with:



& .\.venv\Scripts\python.exe plot_temperature.py

