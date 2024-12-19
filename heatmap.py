import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# Results simply hardcode pulled from basic_linear_classifer --doDraw False

codebert = [[.95, .97, .91, .94],
[.35, .98, .92, .88],
[.96, .36, .93, .92],
[.97, .98, .28, .92],
[.96, .98, .93, .24]]


codet5 = [
[.95, .98, .93, .96],
[.53, .97, .89, .99],
[.98, .13, .84, .98],
[.98, .96, .19,.98],
[.99,.98, .85,.29]
]

codebertL = [[.95, .97, .91, .94],
[0.0, 0.94, 0.0, 0.34],
[0.02, 0.01, 0.01, 0.98],
[1, 0, 0, 0],
[0.95, 0, 0, 0.19]
]

codet5L = [[.95, .98, .93, .96],
[1, 0, 0, 0.01],
[1, 0, 0, 0],
[1, 0, 0, 0],
[0.99, 0, 0, 0.06]
]

aa = np.array(codet5L)

from matplotlib import colormaps

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    im = ax.imshow(data, cmap='viridis', **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("white", "black"),
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

fig, ax = plt.subplots()

im, cbar = heatmap(aa, ["None", "Python", "Java", "Javascript", "PHP"], ["Python", "Java", "Javascript", "PHP"])
texts = annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
plt.savefig('codet5L_heatmap.png')
plt.show()

