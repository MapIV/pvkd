from labels_colors import labels, color_map

def PlotColorPalette(category_list, color_list):
    """!
    Visualize the color palette
    @param category_list: list of categories to plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    data_color = np.empty((len(category_list), 3))
    data_label, data_x, data_y = [], [], []
    x_ticks, y_ticks = [], [0.5]
    for i, category in enumerate(category_list):
        x_ticks.append(i + 0.5)
        data_label.append(category)
        data_x.append(i + 0.5), data_y.append(0.5)
        data_color[i, :] = np.array([color_list[i][0]/255, color_list[i][1]/255, color_list[i][2]/255]).reshape(1, 3)
    ax.scatter(data_x, data_y, c=data_color, marker='s', s=200)
    ax.axis('square')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(category_list))
    ax.set_xticks(x_ticks)
    plt.xticks(rotation=-90)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(data_label, minor=False)
    ax.set_yticklabels(['prediction'], minor=False)
    plt.title('Category Colormap')
    plt.tight_layout()
    plt.show()

all_labels = []
all_colors = []
keys = [0, 10, 11, 30, 18, 40, 49, 72, 50, 52, 70, 20]
for key in labels:
    if key in keys:
        all_labels.append(labels[key])
        all_colors.append(color_map[key])
PlotColorPalette(all_labels, all_colors)