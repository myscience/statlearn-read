'''
    In this script we augment the original dataset with some additional
    affine transformations such as: translations, rotations and scaling.
'''

import numpy as np
from PIL import Image
from tqdm import tqdm

def transform (img, par, size = None, resample = Image.BICUBIC):
    # Readout transformation parameters
    tht = par['tht']; trn = par['trn']; scl = par['scl']

    trn  = np.array (trn)
    hshp = np.array (np.shape (img)[:2])[::-1] / 2

    # Compute image center coordinate and new center based on translation
    x, y  = hshp 
    nx, ny = hshp + trn if size is None else np.array (size) / 2 + trn

    sx, sy = scl
    
    tht = np.radians(tht)
    cos = np.cos(tht)
    sin = np.sin(tht)

    a =  cos / sx; b = sin / sx; c = x - nx * a - ny * b
    d = -sin / sy; e = cos / sy; f = y - nx * d - ny * e

    mat = (a, b, c, d, e, f)

    timg = img.transform (img.size if size is None else size, Image.AFFINE, mat, resample = resample) 

    # Check whether transformation resulted in out-of-bounds error
    pix = np.asarray (timg)[..., 0]

    guard = ((pix[:, (0, -1)] == 0).any() or (pix[(0, -1), :] == 0).any())

    return timg, guard



loadpath = '../data/dataset0/'

# Import the original words
words = [Image.open (loadpath + f'word_{i}.png') for i in range (5)]

# Here we augment the dataset
size = 10
outsize = (500, 500)
savepath = '../data/augment0/'
for w, word in enumerate (tqdm (words, desc = 'Augmenting Dataset', leave = False)):
    # Produce random transformations
    Tht = np.random.uniform (-45, 45, size = size)
    Trn = np.random.uniform (-100, 100, size = (size, 2))
    Scl = np.random.uniform (0.8, 1.2, size = size)

    [out[0].save (savepath + f'word_{w}_t{t}.png') for t, (tht, trn, scl) in enumerate (zip (Tht, Trn, Scl)) 
        if (out := transform (word, {'tht' : tht, 'trn' : trn, 'scl' : (scl, scl)}, outsize))[1]]