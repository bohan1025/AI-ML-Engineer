import matplotlib.pyplot as plt


def plotSideBySide(img1, img2, title1="Original", title2="Modified"):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 10))
    ax1.imshow(img1, cmap="gray")
    ax1.set_title(title1)
    ax2.imshow(img2, cmap="gray")
    ax2.set_title(title2)
    plt.axis("off")
    plt.show()


def plotImageWithPoints(img, px, py, title="Image with Points"):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img, cmap="gray")
    ax.scatter(px, py, s=1, c="r")
    ax.set_title(title)
    plt.axis("off")
    plt.show()


def AddPlotImageWithPoints(img, px, py, title="Image with Points", figsize=(15, 10)):
    fig, ax = plt.subplots(figsize)
    ax.imshow(img, cmap="gray")
    ax.scatter(px, py, s=1, c="r")
    ax.set_title(title)
    plt.axis("off")


def plotImageAndImageWithPoints(img1, px, py, title="Same Image with Points"):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 10))
    ax1.imshow(img1, cmap="gray")
    ax2.imshow(img1, cmap="gray")
    ax2.scatter(px, py, s=1, c="r")
    ax2.set_title(title)
    plt.axis("off")
    plt.show()


def plotImage(img, title="Image"):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    plt.axis("off")
    plt.show()


def plotImageWithPointsSideBySide(image1, x1, y1, title1, image2, x2, y2, title2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image1, cmap="gray")
    ax1.scatter(x1, y1, c="r", s=10)
    ax1.set_title(title1)
    ax1.axis("off")
    ax2.imshow(image2, cmap="gray")
    ax2.scatter(x2, y2, c="r", s=10)
    ax2.set_title(title2)
    ax2.axis("off")
    plt.show()
