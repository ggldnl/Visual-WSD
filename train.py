import pytorch_lightning as pl
from data import VisualWSDDataModule
from model import CLIPLike
import config
import os


def find_last_checkpoint(directory):

    # Ensure the provided path is a directory
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return None

    # Get all files ending with '.ckpt' recursively
    checkpoint_files = [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if file.endswith('.ckpt')]

    # Check if any checkpoint files were found
    if not checkpoint_files:
        print(f"No '.ckpt' files found in '{directory}'.")
        return None

    # Select the checkpoint with the last name in alphabetical order
    selected_checkpoint = max(checkpoint_files)

    return selected_checkpoint


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
    # data_module.prepare_data()
    # data_module.setup()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="checkpoint_{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
    )

    # Create the trainer saving checkpoints to CHECK_DIR at every epoch
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        # default_root_dir=config.CHECK_DIR,
        # precision=config.PRECISION,
        # callbacks=[checkpoint_callback]
    )

    # Check if we need to restore a checkpoint or to train from scratch
    selected_checkpoint = find_last_checkpoint(str(config.LOGS_DIR))
    if selected_checkpoint:
        print(f"Selected checkpoint: {selected_checkpoint}")
        model = CLIPLike.load_from_checkpoint(selected_checkpoint)
    else:
        print(f"Training from scratch...")
        model = CLIPLike()

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


