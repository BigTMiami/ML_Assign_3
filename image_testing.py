from mnist_data_prep import get_mnist_data_labels
from sklearn.decomposition import PCA, FastICA
import numpy as np
from matplotlib import pyplot as plt


(
    train_data,
    train_labels,
    test_data,
    test_labels,
) = get_mnist_data_labels(scale_data=True)


def show_pca_images(data, n_components, image_count=4):
    figure, axes = plt.subplots(image_count, 2)
    for i in range(image_count):
        image = data[i]
        image = image.reshape((28, 28))
        image = image * 255
        image = image.astype("uint8")
        axes[i, 0].imshow(image, cmap="gray")

    pca = PCA(n_components)
    data_pca = pca.fit_transform(train_data)
    print(f"PCA Components:{pca.n_components_}")

    for i in range(image_count):
        image = pca.inverse_transform(data_pca[i])
        image = image.reshape((28, 28))
        image = image * 255
        image = image.astype("uint8")
        axes[i, 1].imshow(image, cmap="gray")
    plt.show()


show_pca_images(train_data, 0.9)

image_1 = train_data[0]
image_1 = image_1.reshape((28, 28))
image_1 = image_1 * 255
image_1 = image_1.astype("uint8")
plt.imshow(image_1, cmap="gray")
plt.show()

n_components = 58

pca = PCA(n_components)

train_data_pca_58 = pca.fit_transform(train_data)

image_1_reconstructed = pca.inverse_transform(train_data_pca_58[0])
image_1_reconstructed = image_1_reconstructed.reshape((28, 28))
image_1_reconstructed = image_1_reconstructed * 255
image_1_reconstructed = image_1_reconstructed.astype("uint8")
plt.imshow(image_1_reconstructed, cmap="gray")
plt.show()
