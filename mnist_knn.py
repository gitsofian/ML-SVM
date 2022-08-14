# pip install python-mnist
from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

mndata = MNIST(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'mnist'), return_type="numpy")

images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()

fig, axes = plt.subplots(1, 10)
for i, ax in enumerate(axes):
    subset = np.asarray(labels_train == i).nonzero()[0]
    idx = np.random.choice(subset)
    image = images_train[idx].reshape((28, 28))
    label = labels_train[idx]

    ax.imshow(image, cmap="gray")
    ax.set_title(label)
    ax.axis("off")

# Aufgabe a) -->

# <--

# Aufgabe b) -->
pred_test = None
# <--

# Aufgabe c) -->
conf = None
# <--

if pred_test is not None:
    fig, axes = plt.subplots(4, 8)
    fig.suptitle("Random Test Set Examples")
    axes = axes.flatten()
    for ax in axes:
        idx = np.random.randint(0, images_test.shape[0])
        image = images_test[idx].reshape((28, 28))
        label = labels_test[idx]
        pred = pred_test[idx]

        ax.imshow(image, cmap="gray")
        ax.set_title("%d (%d)" % (pred, label))
        ax.axis("off")


if conf is not None:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(conf, cmap="YlGn")
    # ax.axis("off")
    ax.set_xlabel("prediction")
    ax.set_ylabel("ground truth")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    for i in range(10):
        for j in range(10):
            ax.text(j, i, "%d" % conf[i, j], color="black",
                    horizontalalignment='center', verticalalignment='center')


plt.show()
