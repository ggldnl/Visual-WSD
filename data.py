from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import random_split
import pytorch_lightning as pl
import config
import utils
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
                 data_folder,
                 data_file,
                 gold_file,
                 image_transform=None,
                 caption_transform=None,
                 ):

        self.data_folder = data_folder

        with open(data_file, 'r', encoding="utf-8") as file:
            self.data = [(row[0], row[1], row[2:]) for row in csv.reader(file, delimiter='\t')]

        with open(gold_file, 'r', encoding="utf-8") as file:
            self.gold = [row[0] for row in csv.reader(file, delimiter='\t')]

        self.image_transform = image_transform
        self.caption_transform = caption_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label, context, images_list = self.data[index]
        gold = self.gold[index]

        # Gold image
        gold_image_path = os.path.join(self.data_folder, gold)
        gold_image = Image.open(gold_image_path)

        # Caption
        caption = f'{label} {context}'

        # Wrong matches
        # images_list = [Image.open(os.path.join(self.data_folder, image_path)).convert("RGB") for image_path in
        #                images_list]

        if self.image_transform:
            gold_image = self.image_transform(gold_image)

        if self.caption_transform:
            caption = self.caption_transform(caption)

        return gold_image, caption


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

    def setup(self, stage=None):

        # Define transforms pipeline that images and captions will be subjected to
        image_transforms = tt.Compose([
            tt.Resize((256, 256)),
            ct.RGBTransform(),
            tt.ToTensor()
        ])
        caption_transforms = tt.Compose([
            ct.CaptionTransform()
        ])

        # Define train and val datasets
        train_dataset = VisualWSDDataset(
            self.resources['train_images_path'],
            self.resources['train_data_path'],
            self.resources['train_gold_path'],

            # The images will be transformed into tensors by the preprocessor inside the model itself
            image_transform=image_transforms,

            # The captions will be transformed into tensors by the tokenizer inside the model itself
            caption_transform=caption_transforms
        )
        self.train_dataset, self.val_dataset = random_split(train_dataset, [self.train_val_split, 1 - self.train_val_split])

        # Define test dataset
        en_test_dataset = VisualWSDDataset(
            self.resources['test_images_path'],
            self.resources['en_test_data_path'],
            self.resources['en_test_gold_path'],
            image_transform=image_transforms,
            caption_transform=caption_transforms
        )

        it_test_dataset = VisualWSDDataset(
            self.resources['test_images_path'],
            self.resources['it_test_data_path'],
            self.resources['it_test_gold_path'],
            image_transform=image_transforms,
            caption_transform=caption_transforms
        )

        fa_test_dataset = VisualWSDDataset(
            self.resources['test_images_path'],
            self.resources['fa_test_data_path'],
            self.resources['fa_test_gold_path'],
            image_transform=image_transforms,
            caption_transform=caption_transforms
        )

        self.test_dataset = ConcatDataset([
            en_test_dataset,
            it_test_dataset,
            fa_test_dataset
        ])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == '__main__':

    root_folder = r'/home/daniel/Git/CLIP/data/'
    visual_wsd_datamodule = VisualWSDDataModule(root_folder)
    visual_wsd_datamodule.prepare_data()
    visual_wsd_datamodule.setup()

    train_loader = visual_wsd_datamodule.train_dataloader()
