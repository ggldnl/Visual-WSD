from datamodule_interface import CustomDataModuleInterface
import zipfile
import gdown
import os


class VisualWSDDataModule(CustomDataModuleInterface):
    """
    Custom PyTorch Lightning DataModule class to download a Visual WSD dataset.
    The datamodule will download the content at the url only if some of the
    required files are missing.

    Once extracted, the dataset will have the following structure:
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
    └── test.data.v1.1.gold
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
                 root_folder: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 train_val_split: float = 0.8,
                 **kwargs
                 ):
        """
        Initialize the custom datamodule.

        Args:
            root_folder (str): Root folder where the data should be locally stored.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            num_workers (int, optional): Number of subprocesses used for data loading. Defaults to 4.
            train_val_split (float, optional): Train/Split sample percentage. Defaults to 0.8.
        """

        urls = [
            r'https://drive.google.com/file/d/1byX4wpe1UjyCVyYrT04sW17NnycKAK7N/view?usp=sharing',  # Train + trial data
            r'https://drive.google.com/file/d/10vDZsY0EhzvFFR8IF-3P_2ApOF0GIMML/view?usp=share_link',  # Test data
            r'https://drive.google.com/file/d/15ed8TXY9Pzk68_SCooFm7AfkeFtCd16Q/view?usp=sharing'  # Test images resized
        ]

        test_images_path = r'test_images_resized'
        en_test_gold_path = r'en.test.gold.v1.1.txt'
        en_test_data_path = r'en.test.data.v1.1.txt'
        it_test_gold_path = r'it.test.gold.v1.1.txt'
        it_test_data_path = r'it.test.data.v1.1.txt'
        fa_test_gold_path = r'fa.test.gold.txt'
        fa_test_data_path = r'fa.test.data.txt'
        train_images_path = r'semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1'
        train_data_path = r'semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt'
        train_gold_path = r'semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt'

        required_files = [
            test_images_path,
            en_test_gold_path,
            en_test_data_path,
            it_test_gold_path,
            it_test_data_path,
            fa_test_gold_path,
            fa_test_data_path,
            train_images_path,
            train_data_path,
            train_gold_path
        ]

        super().__init__(
            urls,
            required_files,
            root_folder,
            **kwargs
        )

        # Pretty self-explanatory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """
        Initializes the dataset.

        Args:
            stage (str, optional): Stage of training (e.g., 'fit', 'validate', 'test'). Defaults to None.
        """
        if stage == 'fit' or stage is None:
            pass

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return None

    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return None


if __name__ == '__main__':

    root_folder = r'/home/daniel/Git/CLIP/data/'
    visual_wsd_datamodule = VisualWSDDataModule(root_folder)
    visual_wsd_datamodule.prepare_data()
    visual_wsd_datamodule.setup()
