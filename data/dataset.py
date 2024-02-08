from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):

    def __init__(self, data, targets, transform=None):
        """
        Initialize the CustomDataset.

        Args:
            data (list): List of image data (e.g., PIL Images or NumPy arrays).
            targets (list): List of corresponding labels.
            transform (callable, optional): Optional transform to be applied to the image data.
        """

        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Get the item at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image data and its corresponding label.
        """

        image = self.data[index]
        label = self.targets[index]

        # Apply transform if it exists
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':

    # Get a list of images along with their name as label
    from PIL import Image
    import os

    def get_n_images(folder, n=10):
        image_names = []
        image_objects = []

        # Get a list of all files in the folder
        all_files = os.listdir(folder)

        # Filter only the files with supported image extensions (you can customize this list)
        image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if len(image_files) > n:
            raise RuntimeError(f'Not enough images in folder {folder}!')

        for image_file in image_files[:n]:
            image_path = os.path.join(folder_path, image_file)
            image_names.append(image_file)
            image_objects.append(Image.open(image_path))

        return image_names, image_objects

    folder_path = f''
    names, images = get_n_images(folder_path)

    # Define the transforms
    from transforms.rgb_transform import RGBTransform

    image_transforms = transforms.Compose([
        RGBTransform(),
        transforms.Resize((256, 256)),
        # transforms.ToTensor(),
    ])

    test_dataset = CustomDataset(
        images,
        names,
        image_transforms
    )

    for image, label in test_dataset[:10]:
        print(f'{image.mode} mage with size {image.size} captioned: {label}')
