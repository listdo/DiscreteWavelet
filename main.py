from PIL import Image
from numpy import asarray
import pywt.data
import numpy as np
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

# https://pywavelets.readthedocs.io/en/latest/
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
# https://pywavelets.readthedocs.io/en/latest/

# load the image
image = Image.open('kolala_bw.jpeg')

# initial printing
def print_original():
    global original
    # convert image to numpy array
    original = asarray(image)
    print(type(original))
    # summarize shape
    print(original.shape)


# single resolution with or without noise in wavelets
def single_resolution(add_noise = False):

    # Wavelet transform of image
    titles = ['LL', ' LH', 'HL', 'HH']
    coeffs2 = pywt.dwt2(original, 'haar')

    # Coefficients
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))

    coeffArray = [LL, LH, HL, HH]

    for i in range(0, len(coeffArray)):

        a = coeffArray[i]

        # if add_noise add a simple gaussian noise with a sigma of 5
        if add_noise:
            row, col = a.shape
            mean = 1
            sigma = 25
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
            noisy = a + gauss
            a = noisy
            coeffArray[i] = a

        # outputs
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)

        if add_noise:
            ax.set_title(titles[i] + " Noisy", fontsize=10)
        else:
            ax.set_title(titles[i], fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    # plotting
    plt.show()

    myTupel = coeffArray[0], (coeffArray[1], coeffArray[2], coeffArray[3])

    result = pywt.idwt2(myTupel, 'haar')

    fig, ax = plt.subplots()
    ax.set_title("Result")
    ax.imshow(result, interpolation="nearest", cmap=plt.cm.gray)
    plt.show()


def multi_resolution():
    x = original
    shape = x.shape

    # amount of levels
    max_lev = 3

    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    for level in range(0, max_lev + 1):
        if level == 0:
            # show original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(x, cmap=plt.cm.gray)
            axes[1, 0].set_title('Image')
            axes[1, 0].set_axis_off()
            continue

        # plot boundaries
        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level], label_levels=max_lev)
        axes[0, level].set_title('{} level\ndecomposition'.format(level))

        # compute the 2D DWT
        c = pywt.wavedec2(x, 'haar', level=level)

        # normalize
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d / np.abs(d).max() for d in c[detail_level + 1]]

        # normalized coefficients
        arr, slices = pywt.coeffs_to_array(c)

        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
        axes[1, level].set_axis_off()
    plt.tight_layout()
    plt.show()


print_original()
single_resolution()
multi_resolution()
single_resolution(add_noise=True)