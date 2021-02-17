'''
    In this source we build the dataset used in the Statistical Visual
    Leaning work. The idea is to compose a set of words made up of
    pseudo-characters. Words come in two flavours: standard and deviants.
    They differ based on their character frequency. Here we produce a
    dataset by randomly assemble pseudo-words based on such statistics.
'''

# Import library for image manipulation
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

chr_path = '../data/Characters/'

# These are idx for standard and deviant character extraction
word_idx = [[1,2,4],
            [5,2,3],
            [1,6,3],
            [7,8,4],
            [5,8,9],
            [7,6,9]]

hfd_idx = [1, 2, 3]
lfd_idx  = [5, 6, 4]

# Import the characters from file
tot = 24
chars = np.array ([Image.open (chr_path + f'{idx}.jpg') for idx in range(1, tot)], dtype = np.object)

# Here we identify the largest character width and height
shapes = np.array ([np.shape(char)[:2] for char in chars])

max_h = max (shapes[:, 0])
max_w = max (shapes[:, 1])

# We can now compute appropriate word size
# NOTE: This is needed for a ANN which needs consistent input dimensions
wrd_size = (3 * max_w, max_h)

dataset = f'../data/dataset{0}/'
chr_perm = np.random.permutation (tot - 1)
chr_idxs = [chr_perm[idx] for idx in word_idx]

# Extract random characters for words and high-low frequency deviants
chr_imgs_set = [chars[idx] for idx in chr_idxs]
chr_hfd, chr_lfd  = chars[hfd_idx], chars[lfd_idx]

def white2alpha(img, thr = (100, 100, 100, 100)):
    # Mask white pixels to alpha channel
    pix = np.array (img)
    mask = pix > thr

    # Set alpha value to full-transparency
    pix [mask[..., 0]] = 0
    
    return Image.fromarray (pix)

# Create new empty 3-chars word image
for i, chr_imgs in enumerate (chr_imgs_set):
    word = Image.new ('RGBA', wrd_size)

    # Get the character width for appropriate positioning
    chrs_h = [np.shape (chr_imgs)[0] for chr_imgs in chr_imgs]
    chrs_w = [np.shape (chr_imgs)[1] for chr_imgs in chr_imgs]

    wrd_w = sum (chrs_w)
    pad = (max_w * 3 - wrd_w) // 2

    # dims = np.array ([0] + [np.shape (chr_img)[1] for chr_img in chr_imgs])
    dims = [(w, h) for w, h in zip(np.cumsum ([pad] + chrs_w)[:-1], chrs_h)]

    # Here we assemble the word char by char
    [word.paste (chr_img, at) for chr_img, at in zip(chr_imgs, ((w, max_h - h) for w, h in dims))]

    # Convert white pixels to alpha
    word = white2alpha(word)

    # Here we save the word in current dataset
    word.save(dataset + f'word_{i}.png')


# Create & Store the high-low frequency deviants
hfd_word = Image.new('RGBA', wrd_size)    
lfd_word = Image.new('RGBA', wrd_size)    

hfd_chars = chars[chr_perm[hfd_idx]]
lfd_chars = chars[chr_perm[lfd_idx]]

# Get the character width for appropriate positioning
hfd_chrs_h = [np.shape (chr_imgs)[0] for chr_imgs in hfd_chars]
hfd_chrs_w = [np.shape (chr_imgs)[1] for chr_imgs in hfd_chars]

lfd_chrs_h = [np.shape (chr_imgs)[0] for chr_imgs in lfd_chars]
lfd_chrs_w = [np.shape (chr_imgs)[1] for chr_imgs in lfd_chars]

hfd_wrd_w = sum (hfd_chrs_w)
hfd_pad = (max_w * 3 - hfd_wrd_w) // 2

lfd_wrd_w = sum (lfd_chrs_w)
lfd_pad = (max_w * 3 - lfd_wrd_w) // 2

# dims = np.array ([0] + [np.shape (chr_img)[1] for chr_img in chr_imgs])
hfd_dims = [(w, h) for w, h in zip(np.cumsum ([hfd_pad] + hfd_chrs_w)[:-1], hfd_chrs_h)]
lfd_dims = [(w, h) for w, h in zip(np.cumsum ([lfd_pad] + lfd_chrs_w)[:-1], lfd_chrs_h)]

[hfd_word.paste (chr_img, at) for chr_img, at in zip(hfd_chars, ((w, max_h - h) for w, h in hfd_dims))]
[lfd_word.paste (chr_img, at) for chr_img, at in zip(lfd_chars, ((w, max_h - h) for w, h in lfd_dims))]

# Convert white pixels to alpha
hfd_word = white2alpha (hfd_word)
lfd_word = white2alpha (lfd_word)

hfd_word.save(dataset + f'hfd_word.png')
lfd_word.save(dataset + f'lfd_word.png')