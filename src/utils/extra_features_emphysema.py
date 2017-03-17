from preprocessing import *
from scipy import stats
import matplotlib.pyplot as plt

threshold = -250


def get_emphysema_predictors(img, mask):

    pix_1d = np.ndarray.flatten(img)
    mask_1d = np.ndarray.flatten(mask)
    pix_lung = pix_1d[mask_1d == 1]
    gated_pix_lung = pix_lung[pix_lung < threshold]

    # the histogram of the data
    # plt.hist(pix_lung, 50, normed=1, facecolor='green', alpha=0.75)

    gated_skewness = stats.skew(gated_pix_lung)
    gated_kurtosis = stats.kurtosis(gated_pix_lung)

    return gated_skewness, gated_kurtosis


def compute_emphysema_probability(img, mask):

    probability = 0

    gated_skewness, gated_kurtosis = get_emphysema_predictors(img, mask)

    #TODO: Combine gated_skewness, gated_kurtosis to obtain an estimate of the probability of the emphysema

    return probability
