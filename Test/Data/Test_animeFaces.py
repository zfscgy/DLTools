import matplotlib.pyplot as plt
from Data import AnimeFacesLoader


def test_anime_faces():
    dataloader = AnimeFacesLoader((224, 224))
    xs = dataloader.get_train_batch(3)
    for i in range(3):
        plt.imshow(xs[i])
        plt.show()


test_anime_faces()