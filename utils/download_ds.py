from urllib import request
import os
import subprocess
import zipfile


os.environ['KAGGLE_USERNAME'] = "szymonindy"
os.environ['KAGGLE_KEY'] = "cd877f5229aaec7124d414790f0830e5"
subprocess.run("pip install kaggle")
subprocess.run("kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")

with zipfile.ZipFile('chest-xray-pneumonia.zip', 'r') as zip_ref:
    zip_ref.extractall('.')