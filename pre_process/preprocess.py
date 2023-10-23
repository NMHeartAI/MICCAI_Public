import skimage
from scipy.ndimage import gaussian_filter
from skimage import exposure

from utils.DDFB import *
from utils.guided_filter import *
from utils.homomorphic_filter import *
from utils.multiscale_tophat import *

def preprocess(img):

    homo = HomomorphicFilter()
    radius = 8
    eps = 0.2
    alpha = 2

    # Define the scale parameter sigma
    sigma = 2

    image = img.copy()

    if len(image.shape) > 2:
        p('image different shape: {}'.format(image.shape))
        image = image.mean(axis=np.argmin(image.shape))
        p('image new shape: {}'.format(image.shape))

    # 1 homomorphic transform
    homomorphic_img = homo.apply_filter(image.copy())
    
    # 2 Normalize
    nml_homomorphic_img = normalize(homomorphic_img)
   
    return nml_homomorphic_img
