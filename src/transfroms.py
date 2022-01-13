import numpy as np


class Normalize:
    def __init__(self) -> None:
        pass

    def __call__(self, image, mask=None, **kwargs):
        image = (image / 255 - 0.45) / 0.22
        out = {"image": image, "mask": mask}
        return out


class RandomCrop:
    def __init__(self, crop_size) -> None:
        self.crop_size = crop_size

    def __call__(self, image, mask=None, **kwargs):

        w, h = image.shape[0], image.shape[1]

        wi = np.random.randint(0, w - self.crop_size[0])
        hi = np.random.randint(0, h - self.crop_size[1])

        image = image[wi : wi + self.crop_size[0], hi : hi + self.crop_size[1]]
        mask = mask[wi : wi + self.crop_size[0], hi : hi + self.crop_size[1]]

        out = {"image": image, "mask": mask}

        return out


class HardCutout:
    def __init__(self, crop_size, p) -> None:
        self.crop_size = crop_size
        self.p = p

    def __call__(self, image, mask=None, **kwargs):

        if np.random.rand() < self.p:

            w, h = image.shape[0], image.shape[1]

            wi = np.random.randint(0, w - self.crop_size[0])
            hi = np.random.randint(0, h - self.crop_size[1])

            image[:wi, :] = 0
            image[:, :hi] = 0
            image[wi + self.crop_size[0] :, :] = 0
            image[:, hi + self.crop_size[1] :] = 0

            mask[:wi, :] = 0
            mask[:, :hi] = 0
            mask[wi + self.crop_size[0] :, :] = 0
            mask[:, hi + self.crop_size[1] :] = 0

        out = {"image": image, "mask": mask}

        return out
