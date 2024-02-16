from typing import Callable


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
