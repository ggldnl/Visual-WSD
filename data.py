from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
import numpy as np
import config
import utils
import torch
import csv
import os

# Torchvision transforms
from torchvision import transforms as tt

# Custom transforms
import transforms as ct

# Import PIL and enable image truncation
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VisualWSDDataset(Dataset):

    def __init__(self,
                 data,
                 image_transform=None,
                 caption_transform=None,
                 ):

        self.data = data

        self.image_transform = image_transform
        self.caption_transform = caption_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        caption, image_path = self.data[index]
        image = Image.open(image_path).convert('RGB')

        if self.image_transform:
            image = self.image_transform(image)

        if self.caption_transform:
            caption = self.caption_transform(caption)

        return image, caption


class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch):
        x, y = zip(*batch)
        return torch.stack(x), list(y)


class VisualWSDDataModule(pl.LightningDataModule):
    """
    Custom PyTorch Lightning DataModule class to download a Visual WSD dataset.

    Once downloaded and extracted, the dataset will have the following structure:
    root_folder
    ├── semeval-2023-task-1-V-WSD-train-v1
    │         ├── README
    │         ├── train_v1
    │         │         ├── train_images_v1
    │         │         │       (images...)
    │         │         ├── train.data.v1.txt
    │         │         └── train.gold.v1.txt
    │         └── trial_v1
    │             ├── trial_images_v1
    │             │       (images...)
    │             ├── trial.data.v1.txt
    │             └── trial.gold.v1.txt
    ├── en.test.data.v1.1.txt
    ├── en.test.gold.v1.1.txt
    ├── fa.test.data.txt
    ├── fa.test.gold.txt
    ├── it.test.data.v1.1.txt
    ├── it.test.gold.v1.1.txt
    └── README.txt

    More information here on the dataset here:
    https://raganato.github.io/vwsd/
    """

    def __init__(self,
                 data_dir,
                 download=False,
                 train_val_split=0.8,
                 batch_size=config.BATCH_SIZE,
                 num_workers=config.NUM_WORKERS,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.download = download
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.urls = [
            r'https://drive.google.com/file/d/1byX4wpe1UjyCVyYrT04sW17NnycKAK7N/view?usp=sharing',  # Train + trial data
            r'https://drive.google.com/file/d/10vDZsY0EhzvFFR8IF-3P_2ApOF0GIMML/view?usp=share_link',  # Test data
            r'https://drive.google.com/file/d/15ed8TXY9Pzk68_SCooFm7AfkeFtCd16Q/view?usp=sharing'  # Test images resized
        ]

        # Required files
        self.resources = {
            'test_images_path': r'test_images_resized',
            'en_test_gold_path': r'en.test.gold.v1.1.txt',
            'en_test_data_path': r'en.test.data.v1.1.txt',
            'it_test_gold_path': r'it.test.gold.v1.1.txt',
            'it_test_data_path': r'it.test.data.v1.1.txt',
            'fa_test_gold_path': r'fa.test.gold.txt',
            'fa_test_data_path': r'fa.test.data.txt',
            'train_images_path': r'semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1',
            'train_data_path': r'semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt',
            'train_gold_path': r'semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt',
        }

        # Make the path absolute for each path in the resource dict
        self.resources = {
            key: os.path.join(data_dir, value) for key, value in self.resources.items()
        }

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.collate_fn = CollateFn()

    def prepare_data(self):

        if self.download:

            # Create the folder if it does not exist
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)

            for url in self.urls:

                # Download the resource in the data directory with name data.zip
                resource_filename = os.path.join(self.data_dir, "data.zip")
                utils.download_google_drive(url, resource_filename)

                # Extract the resource
                utils.extract_resource(resource_filename, self.data_dir)

                # Remove the zip file
                os.remove(resource_filename)

    @staticmethod
    def get_data(images_path, data_path, gold_path):

        # label, caption, list of images
        with open(data_path, 'r', encoding="utf-8") as file:
            data = [(row[0], row[1], row[2:]) for row in csv.reader(file, delimiter='\t')]

        # gold image
        with open(gold_path, 'r', encoding="utf-8") as file:
            gold = [row[0] for row in csv.reader(file, delimiter='\t')]

        captions = [f'{label} {context}' for label, context, _ in data]
        gold_images = [os.path.join(images_path, gold_image) for gold_image in gold]

        return np.column_stack((captions, gold_images))

    def split(self, data, percentages):

        if isinstance(percentages, float) and 0 < percentages < 1:
            return self._split_two(data, [percentages, 1 - percentages])

        if isinstance(percentages, list):

            assert sum(percentages) == 1, f'Elements in the percentages array does not sum up to 1: {percentages}'

            if len(percentages) == 2:
                return self._split_two(data, percentages)
            elif len(percentages) == 3:
                return self._split_three(data, percentages)
            else:
                raise ValueError(f'Invalid percentages array: {percentages}')

    @staticmethod
    def _split_two(data, percentages):

        # Compute the sizes of each set
        num_samples = len(data)
        num_train = int(percentages[0] * num_samples)
        num_val = int(percentages[1] * num_samples)

        # Shuffle the indices to randomly select samples for each set
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split the indices into training and validation
        train_indices = indices[:num_train]
        val_indices = indices[num_train: num_train + num_val]

        return data[train_indices], data[val_indices]

    @staticmethod
    def _split_three(data, percentages):

        # Compute the sizes of each set
        num_samples = len(data)
        num_train = int(percentages[0] * num_samples)
        num_val = int(percentages[1] * num_samples)

        # Shuffle the indices to randomly select samples for each set
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split the indices into training and validation
        train_indices = indices[:num_train]
        val_indices = indices[num_train: num_train + num_val]
        test_indices = indices[num_train + num_val:]

        return data[train_indices], data[val_indices], data[test_indices]

    def setup(self, stage=None):

        # Define transforms pipeline that images and captions will be subjected to
        image_transforms = tt.Compose([
            tt.Resize((256, 256)),
            tt.ToTensor()
        ])
        caption_transforms = tt.Compose([
            ct.CaptionTransform()
        ])

        train_val_data = self.get_data(
            self.resources['train_images_path'],
            self.resources['train_data_path'],
            self.resources['train_gold_path']
        )

        train_data, val_data = self.split(train_val_data, [self.train_val_split, 1 - self.train_val_split])

        # Define train and val datasets
        self.train_dataset = VisualWSDDataset(
            train_data,

            # The images will be transformed into tensors by the preprocessor inside the model itself
            image_transform=image_transforms,

            # The captions will be transformed into tensors by the tokenizer inside the model itself
            caption_transform=caption_transforms
        )

        self.val_dataset = VisualWSDDataset(
            val_data,
            image_transform=image_transforms,
            caption_transform=caption_transforms
        )

        en_test_data = self.get_data(
            self.resources['test_images_path'],
            self.resources['en_test_data_path'],
            self.resources['en_test_gold_path'],
        )

        # Define test dataset
        en_test_dataset = VisualWSDDataset(
            en_test_data,
            image_transform=image_transforms,
            caption_transform=caption_transforms
        )

        it_test_data = self.get_data(
            self.resources['test_images_path'],
            self.resources['it_test_data_path'],
            self.resources['it_test_gold_path'],
        )

        it_test_dataset = VisualWSDDataset(
            it_test_data,
            image_transform=image_transforms,
            caption_transform=caption_transforms
        )

        fa_test_data = self.get_data(
            self.resources['test_images_path'],
            self.resources['fa_test_data_path'],
            self.resources['fa_test_gold_path'],
        )

        fa_test_dataset = VisualWSDDataset(
            fa_test_data,
            image_transform=image_transforms,
            caption_transform=caption_transforms
        )

        self.test_dataset = ConcatDataset([
            en_test_dataset,
            it_test_dataset,
            fa_test_dataset
        ])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )


if __name__ == '__main__':

    root_folder = r'dataset'
    visual_wsd_datamodule = VisualWSDDataModule(root_folder)
    visual_wsd_datamodule.prepare_data()
    visual_wsd_datamodule.setup()

    train_loader = visual_wsd_datamodule.train_dataloader()

    # Iterate through the DataLoader
    for batch in train_loader:
        x_batch, y_batch = batch
        print("First column of the batch contains stacked images converted to tensors:\n", x_batch)
        print("Second column of the batch contains stacked strings:\n", y_batch)
        break
