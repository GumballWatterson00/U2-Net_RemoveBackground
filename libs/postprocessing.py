import logging
from PIL import Image

logger = logging.getLogger(__name__)

def run():
    return RemovingTooTransparentBordersHardAndBlurringHardBorders()

class RemovingTooTransparentBordersHardAndBlurringHardBorders:

    def __init__(self):
        import cv2
        import skimage
        import numpy as np
        self.cv2 = cv2
        self.skimage = skimage
        self.np = np

        self.model = None
        self.prep_image = None
        self.orig_image = None

    @staticmethod
    def __extact_alpha_channel__(image):
        """
        Extracts alpha channel from RGBA image
        :param image: RGBA pil image
        :return: RGB Pil image
        """
        # Extract just the alpha channel
        alpha = image.split()[-1]
        # Create a new image with an opaque black background
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        # Copy the alpha channel to the new image using itself as the mask
        bg.paste(alpha, mask=alpha)
        return bg.convert("RGB")

    def __blur_edges__(self, imaged):
        """
        Blurs the edges of the image
        :param imaged: RGBA Pil image
        :return: RGBA PIL  image
        """
        image = self.np.array(imaged)
        image = self.cv2.cvtColor(image, self.cv2.COLOR_RGBA2BGRA)
        # extract alpha channel
        a = image[:, :, 3]
        # blur alpha channel
        ab = self.cv2.GaussianBlur(a, (0, 0), sigmaX=2, sigmaY=2, borderType=self.cv2.BORDER_DEFAULT)
        # stretch so that 255 -> 255 and 127.5 -> 0
        # noinspection PyUnresolvedReferences
        aa = self.skimage.exposure.rescale_intensity(ab, in_range=(140, 255), out_range=(0, 255))
        # replace alpha channel in input with new alpha channel
        out = image.copy()
        out[:, :, 3] = aa
        image = self.cv2.cvtColor(out, self.cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(image)

    def __remove_too_transparent_borders__(self, mask, tranp_val=31):
        mask = self.np.array(mask.convert("L"))
        height, weight = mask.shape
        for h in range(height):
            for w in range(weight):
                val = mask[h, w]
                if val > tranp_val:
                    mask[h, w] = 255
                else:
                    mask[h, w] = 0
        return Image.fromarray(mask)

    def run(self, _, image, orig_image):
        """
        Runs an image post-processing algorithm to improve background removal quality.
        :param _: The class of the neural network used to remove the background.
        :param image: Image without background
        :param orig_image: Source image
        """
        mask = self.__remove_too_transparent_borders__(self.__extact_alpha_channel__(image))
        empty = Image.new("RGBA", orig_image.size)
        image = Image.composite(orig_image, empty, mask)
        image = self.__blur_edges__(image)
        return image
