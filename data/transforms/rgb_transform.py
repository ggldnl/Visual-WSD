from typing import Callable


class RGBTransform(Callable):

    def __init__(self):
        """
        Initialize the RGBTransform.
        """

        super().__init__()

    def __call__(self, image):
        """
        Apply the transformation to the input image.

        Args:
            image (PIL Image): A PIL Image.

        Returns:
            image (PIL Image): the transformed image with three separate RGB channels.
        """

        return image.convert("RGB")


if __name__ == '__main__':

    # Define an image of a dog
    url = f"https://unsplash.com/photos/Siuwr3uCir0/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8NHx8YmVhY2h8fDB8fHx8MTYzNTg0MjYzMg&w=640"

    # Load the image
    from PIL import Image
    import requests

    def load_image(url_or_path):
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            return Image.open(requests.get(url_or_path, stream=True).raw)
        else:
            return Image.open(url_or_path)

    image_rgba = load_image(url).convert('RGBA')

    # Apply the ConvertToRGB transform
    converter = RGBTransform()
    image_rgb = converter(image_rgba)

    # Print shape of the image before and after
    print(image_rgba.mode)
    print(image_rgb.mode)
