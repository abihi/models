from PIL import Image
import numpy as np

from matplotlib import gridspec
from matplotlib import pyplot as plt

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def vis_segmentation(org_image, image, seg_map, opt):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 6])

    plt.subplot(grid_spec[0])
    plt.imshow(org_image)
    plt.axis('off')
    plt.title('Original image')

    plt.subplot(grid_spec[1])
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original labeled image')

    plt.subplot(grid_spec[2])
    if opt == 1:
        seg_image = seg_map
    else:
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('relabeled image')

    plt.subplot(grid_spec[3])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.5)
    plt.axis('off')
    plt.title('relabel overlay')

    plt.grid('off')
    plt.show()


LABEL_NAMES = np.asarray([
    'background', 'wall', 'floor'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
