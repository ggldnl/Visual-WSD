from typing import Callable


class RGBTransform(Callable):

    def __init__(self):

        super().__init__()

    def __call__(self, image):
        """
        Apply the transformation to the input image

        Args:
            image (PIL Image): A PIL Image

        Returns:
            image (PIL Image): the transformed image with three separate RGB channels
        """

        return image.convert("RGB")


class CaptionTransform(Callable):

    def __init__(self):

        super().__init__()

    def __call__(self, caption):
        """
        Apply the transformation to the input caption

        Args:
            caption (str): A caption string

        Returns:
            caption (str): A new caption with additional knowledge scraped from wordnet and babelnet
        """

        # TODO
        return caption
