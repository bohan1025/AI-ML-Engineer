import matplotlib.pyplot as plt


def plot_votes(votes_coord, img, img_name):
    colors = {1: "g", 2: "b", 3: "r", 4: "r"}

    plt.imshow(img, cmap="gray")
    for i in range(len(votes_coord)):
        votes_coord_i_x, votes_coord_i_y = votes_coord[i].T
        plt.scatter(votes_coord_i_x, votes_coord_i_y, s=1, c=colors[i + 1])
    plt.title(f"{img_name} votes (red = 3+, blue = 2, green = 1)")
    plt.axis("off")

    plt.show()
