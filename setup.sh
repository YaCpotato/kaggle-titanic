python -m venv .
source bin/activate
pip install -U pip
pip install kaggle
kaggle install titanic
rm titanic.zip && unzip titanic
