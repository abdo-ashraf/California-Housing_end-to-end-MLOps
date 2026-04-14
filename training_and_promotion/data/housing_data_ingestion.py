from pathlib import Path
import urllib.request
import tarfile
import pandas as pd


def download_housing_dataset():
    """Download and extract the California housing dataset if it is missing."""
    base_dir = Path(__file__).resolve().parent.parent.parent

    datasets_dir = base_dir / "assets" / "datasets"
    temp_dir = base_dir / "assets" / "temp"
    tarball_path = temp_dir / "housing.tgz"
    csv_path = datasets_dir / "housing" / "housing.csv"


    datasets_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        url = "https://github.com/ageron/data/raw/main/housing.tgz"

        # Download tarball
        urllib.request.urlretrieve(url, tarball_path)

        # Extract
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=datasets_dir, filter="data")

        # Remove tar file after extraction
        tarball_path.unlink()

    return csv_path


def load_housing_dataset(csv_path):
    return pd.read_csv(csv_path)


# Backward-compatible aliases for older notebooks/scripts.
download_housing_data = download_housing_dataset
load_housing_data = load_housing_dataset
