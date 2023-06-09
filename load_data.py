import os

import gdown
import zipfile
import shutil


if __name__ == "__main__":
    data_url = 'https://drive.google.com/drive/folders/1ols8XxXcrq6ogfkPMMbS6zx8XycgwEyR?usp=sharing'
    data_file = 'data_and_saves/data.zip'

    gdown.download_folder(data_url, quiet=True, use_cookies=False)

    with zipfile.ZipFile(data_file, "r") as f:
        f.extractall(".")

    shutil.rmtree("data_and_saves")

