import pytorch_lightning as pl
from data import VisualWSDDataModule
from model import CLIPLike
import config
import os


if __name__ == '__main__':

    # Define paths
    root_folder = '/home/daniel/Git/Hot-Topics-in-NLP/notebooks/homeworks/homework_4/data'

    test_images_path = os.path.join(root_folder, r'test_images_resized')

    en_test_gold_path = os.path.join(root_folder, r'test.data.v1.1.gold/en.test.gold.v1.1.txt')
    en_test_data_path = os.path.join(root_folder, r'test.data.v1.1.gold/en.test.data.v1.1.txt')
    it_test_gold_path = os.path.join(root_folder, r'test.data.v1.1.gold/it.test.gold.v1.1.txt')
    it_test_data_path = os.path.join(root_folder, r'test.data.v1.1.gold/it.test.data.v1.1.txt')
    fa_test_gold_path = os.path.join(root_folder, r'test.data.v1.1.gold/fa.test.gold.txt')
    fa_test_data_path = os.path.join(root_folder, r'test.data.v1.1.gold/fa.test.data.txt')

    train_images_path = os.path.join(root_folder, r'semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1')

    train_data_path = os.path.join(root_folder, r'semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt')
    train_gold_path = os.path.join(root_folder, r'semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt')

    # Create a datamodule
    data_module = VisualWSDDataModule('dataset', download=False)
    data_module.prepare_data()
    data_module.setup()

    # Create the model
    model = CLIPLike()

    # Create the trainer saving checkpoints to CHECK_DIR at every epoch
    trainer = pl.Trainer(max_epochs=config.EPOCHS, default_root_dir=config.CHECK_DIR)

    # Train the model
    trainer.fit(model, data_module)
