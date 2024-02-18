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

    # trainer.fit(model, data_module)
    # trainer.test(model, data_module)

    """
    Inference
    """

    from PIL import Image
    from tqdm import tqdm
    import csv

    class CustomDatasetLookalike:
        """
        This class mimics a torch Dataset
        """

        def __init__(self,
                     data_folder,
                     data_file,
                     gold_file
                     ):
            self.data_folder = data_folder

            with open(data_file, 'r', encoding="utf-8") as file:
                self.data = [(row[0], row[1], row[2:]) for row in csv.reader(file, delimiter='\t')]

            with open(gold_file, 'r', encoding="utf-8") as file:
                self.gold = [row[0] for row in csv.reader(file, delimiter='\t')]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            label, context, images_names = self.data[index]
            gold_image_name = self.gold[index]

            # Gold image
            gold_image_path = os.path.join(self.data_folder, gold_image_name)
            gold_image = Image.open(gold_image_path).convert("RGB")

            # Wrong matches
            images_list = [Image.open(os.path.join(self.data_folder, image_name)).convert("RGB") for image_name in
                           images_names]

            return {
                'gold_image': gold_image,
                'gold_image_name': gold_image_name,
                'context': context,
                'label': label,
                'images_list': images_list,
                'images_names': images_names
            }

    # Define paths
    root_folder = 'dataset'

    test_images_path = os.path.join(root_folder, r'test_images_resized')
    en_test_gold_path = os.path.join(root_folder, r'en.test.gold.v1.1.txt')
    en_test_data_path = os.path.join(root_folder, r'en.test.data.v1.1.txt')
    it_test_gold_path = os.path.join(root_folder, r'it.test.gold.v1.1.txt')
    it_test_data_path = os.path.join(root_folder, r'it.test.data.v1.1.txt')
    fa_test_gold_path = os.path.join(root_folder, r'fa.test.gold.txt')
    fa_test_data_path = os.path.join(root_folder, r'fa.test.data.txt')

    en_test_dataset = CustomDatasetLookalike(
        test_images_path,
        en_test_data_path,
        en_test_gold_path
    )

    predictions = []
    targets = []
    for elem in tqdm(en_test_dataset):

        images_list = elem['images_list']
        images_names = elem['images_names']
        context = elem['context']
        gold_image_name = elem['gold_image_name']

        targets.append(gold_image_name)

        top_prediction = model.top_k_images(context, images_list, image_paths=images_names)[0]

        predictions.append(top_prediction)

    corrects = sum(x == y for x, y in zip(targets, predictions))
    accuracy = corrects * 100.0 / len(targets)

    print(f'Predicted {corrects}/{len(targets)}\nAccuracy: {accuracy}')
