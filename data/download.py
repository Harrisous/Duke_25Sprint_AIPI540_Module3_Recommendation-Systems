# download movielens-100k dataset to data/raw

import pathlib
import requests
import shutil

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()


def download_movielens_100k():
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    response = requests.get(url)

    if response.status_code == 200:
        # create temp directory
        temp_data_dir = CURRENT_DIR / "temp"
        temp_data_dir.mkdir(parents=True, exist_ok=True)

        # save the zip file to temp directory
        with open(temp_data_dir / "movielens-100k.zip", "wb") as f:
            f.write(response.content)

        # unzip to raw
        import zipfile

        with zipfile.ZipFile(temp_data_dir / "movielens-100k.zip", "r") as zip_ref:
            zip_ref.extractall(CURRENT_DIR / "raw")

        # remove temp directory
        shutil.rmtree(temp_data_dir)

        print("Downloaded movielens-100k dataset successfully.")
    else:
        print("Failed to download the dataset.")


if __name__ == "__main__":
    download_movielens_100k()
