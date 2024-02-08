from torch.utils.data import Dataset
from PIL import Image
import csv
import os


TOKENIZED_SEQUENCE_LEN = 32


class CustomDataset(Dataset):

    def __init__(self,
                 data_folder,
                 data_file,
                 gold_file,
                 image_preprocessor,
                 text_tokenizer,
                 image_transform,
                 text_transform,
                 ):

        super().__init__()

        self.data_folder = data_folder

        with open(data_file, 'r') as file:
            self.data = [(row[0], row[1], row[2:]) for row in csv.reader(file, delimiter='\t')]

        with open(gold_file, 'r') as file:
            self.gold = [row[0] for row in csv.reader(file, delimiter='\t')]

        self.data = self.data[:10]
        self.gold = self.gold[:10]

        self.image_preprocessor=image_preprocessor
        self.text_tokenizer=text_tokenizer

        self.image_transform = image_transform
        self.text_transform = text_transform

    def __len__(self):
        return len(self.gold)

    def __getitem__(self, index):

        label, context, images_list = self.data[index]
        gold = self.gold[index]

        # Gold image retrieval and transform
        gold_image_path = os.path.join(self.data_folder, gold)
        gold_image = Image.open(gold_image_path).convert("RGB")
        if self.image_transform is not None:
            gold_image = self.image_transform(gold_image)

        # Text transform
        caption = context
        if self.text_transform is not None:
            caption = self.text_transform(caption)

        # Apply specific preprocessing for ViT and Bert
        gold_image = self._preprocess_image(gold_image)
        caption = self._preprocess_caption(caption)

        return gold_image, caption

    def _preprocess_image(self, image):
        if self.image_preprocessor is not None:
            inputs = self.image_preprocessor(images=image, return_tensors="pt")
            return inputs['pixel_values']
        else:
            return image

    def _preprocess_caption(self, caption):
        if self.text_tokenizer is not None:

            inputs = self.text_tokenizer.encode_plus(
                caption,
                max_length=TOKENIZED_SEQUENCE_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return inputs['input_ids']
        else:
            return caption

