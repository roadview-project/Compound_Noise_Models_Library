############################################################################################################

# Simplified fog model from ROADVIEW T3.4
# Employs precalculated attenuation coefficient from D3.4 and the CE measured distribution. Other fog
#   visibility based on Koschmieder

############################################################################################################

import numpy as np

def sigma_extinction(visibility):
    if visibility == 10:
        return np.array([0.2889, 0.2934, 0.2989])
    elif visibility == 20:
        return np.array([0.1457, 0.1481, 0.1504])
    elif visibility == 30:
        return np.array([0.1024, 0.1043, 0.1061])
    elif visibility == 50:
        return np.array([0.0705, 0.0723, 0.0740])
    else:
        return np.array([3/visibility, 3/visibility, 3/visibility])

def sigma_BGR(sigma, res):
    sig_mat = np.empty(shape = res, dtype=np.double)
    sig_mat[0::2, 0::2] = sigma[2]
    sig_mat[1::2, 0::2] = sigma[1]
    sig_mat[0::2, 1::2] = sigma[1]
    sig_mat[1::2, 1::2] = sigma[0]

    return sig_mat

def fog_mask(img, depth, sigma_ext, luminosity):
    transmission = np.exp(-sigma_BGR(sigma_ext, np.shape(img))*depth)
    fog_img = transmission * img + (1- transmission)* luminosity
    return fog_img


def fog_application(img, depth, visibility = 50, luminosity = 1500):
    sigma_ext = sigma_extinction(visibility)

    return fog_mask(img, depth, sigma_ext, luminosity) 