import numpy as np
from PIL import Image
import pathlib


class LFWLoader:
    def __init__(self, split=0.8, crop=None):
        path = "./Data/Datasets/LFW/lfw-deepfunneled"
        self.paths = list(pathlib.Path(path).iterdir())
        self.train_len = int(len(self.paths) * split)
        self.crop = crop

    def _crop_img(self, img: Image.Image):
        if not self.crop:
            return img
        assert self.crop[0] <= img.size[0] and self.crop[1] <= img.size[1], "Crop size illegal"
        nw = np.random.randint(0, img.size[0] - self.crop[0])
        nh = np.random.randint(0, img.size[1] - self.crop[1])
        cropped_img = img.crop((nh, nw, self.crop[1] + nh, self.crop[0] + nw))
        return cropped_img

    def _get_batch(self, batch_size, support_size=2, train=True):
        """
        return a batch of N-to-1 data
        for example: batch_size = 2, support_size = 3, returns
        [ [some picture of A, some picture of A (not same)],
          [some picture of A, some picture of B],
          [some picture of A, some picture of C],
          [some picture of B, some picture of B (not same)],
          [some picture of B, some picture of D],
          [some picture of B, some picture of E]
        ]
        :param batch_size:
        :param support_size:
        :param train:
        :return: [batch_size * support_size, 224, 224, 3], [batch_size * support_size, 224, 224, 3], (range from [0, 1] )
                    [batch_size * support_size] range {0, 1}, 1 for match, 0 for not match
        """
        if train:
            paths = self.paths[:self.train_len]
        else:
            paths = self.paths[self.train_len:]
            if batch_size is None:
                batch_size = len(paths)
        imgs0 = []
        imgs1 = []
        labels = []
        for _ in range(batch_size):
            folders = np.random.choice(paths, support_size, replace=False).tolist()
            same_img = np.random.choice(list(folders[0].iterdir()), 2, replace=False)
            same_img[0] = np.asarray(self._crop_img(Image.open(same_img[0])))
            same_img[1] = np.asarray(self._crop_img(Image.open(same_img[1])))
            imgs0.append(same_img[0])
            imgs1.append(same_img[1])
            labels.append(1)
            for folder in folders[1:]:
                img0 = same_img[0]
                img1path = np.random.choice(list(folder.iterdir()))
                img1 = np.asarray(self._crop_img(Image.open(img1path)))
                imgs0.append(img0)
                imgs1.append(img1)
                labels.append(0)
        return np.array(imgs0).astype(np.float32)/255, \
               np.array(imgs1).astype(np.float32)/255, \
               np.array(labels)

    def get_train_batch(self, batch_size, support_size=2):
        return self._get_batch(batch_size, support_size, train=True)

    def get_test_batch(self, batch_size=None, support_size=2):
        return self._get_batch(batch_size, support_size, train=False)

