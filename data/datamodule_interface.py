import pytorch_lightning as pl
from abc import ABC, abstractmethod
import urllib.request
import tempfile
import zipfile
import gdown
import re
import os


class CustomDataModuleInterface(pl.LightningDataModule, ABC):

    def __init__(self,
                 urls,
                 required_paths,
                 root_folder,
                 **kwargs
                 ):

        super().__init__(**kwargs)
        self.urls = urls
        self.required_paths = required_paths
        self.root_folder = root_folder

    def prepare_data(self):
        """
        This method is intended for data downloading and other one-time operations.
        """
        if not self.check_required_files():
            self.download_and_extract()
            if not self.check_required_files():
                raise FileNotFoundError("Required files are missing even after downloading and extraction.")

    def check_required_files(self):
        """
        Checks if all the required files are in place

        Returns:
            bool: True if all the files are in place, False otherwise.
        """
        print('Checking all the files are in place...')
        for path in self.required_paths:
            absolute_path = os.path.join(self.root_folder, path)
            if not os.path.exists(absolute_path):
                print(f"File {absolute_path} is missing.")
                return False
        return True

    @staticmethod
    def is_google_drive_url(url):
        return 'drive.google.com' in url and '/file/d/' in url

    @staticmethod
    def extract_google_drive_file_id(url):
        pattern = r"/file/d/([-\w]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        else:
            return None

    @staticmethod
    def download_google_drive_url(url, dest):
        print(f"Downloading '{url}' to '{dest}'.")
        file_id = CustomDataModuleInterface.extract_google_drive_file_id(url)
        new_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(new_url, dest, quiet=False)

    @staticmethod
    def download_url(url, dest):
        urllib.request.urlretrieve(url, dest)

    def download_and_extract(self):

        print(f'Downloading dataset...')

        for url in self.urls:

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                if self.is_google_drive_url(url):
                    self.download_google_drive_url(url, temp_file.name)
                else:
                    self.download_url(url, temp_file.name)

            try:
                with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                    zip_ref.extractall(self.root_folder)
            except zipfile.BadZipFile:
                print("Unable to extract the file:", url)

    @abstractmethod
    def setup(self, stage=None):
        """
        This method is intended for data loading, splitting datasets, etc.
        """
        raise NotImplementedError("You need to implement the setup() method.")

    @abstractmethod
    def train_dataloader(self):
        """
        DataLoader for training set.
        """
        raise NotImplementedError("You need to implement the train_dataloader() method.")

    @abstractmethod
    def val_dataloader(self):
        """
        DataLoader for validation set.
        """
        raise NotImplementedError("You need to implement the val_dataloader() method.")

    @abstractmethod
    def test_dataloader(self):
        """
        DataLoader for test set.
        """
        raise NotImplementedError("You need to implement the test_dataloader() method.")
