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

# Impose common size for consistency 
# NOTE: This is needed for a ANN which needs consistent input dimensions
chr_size = (620, 574)
wrd_size = (1860, 574)

# Import the characters from file
tot = 24
chars = np.array ([Image.open (chr_path + f'{idx}.jpg').resize(chr_size) for idx in range(1, tot)], dtype = np.object)

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

    # Here we assemble the word char by char
    [word.paste (chr_img, at) for chr_img, at in zip(chr_imgs, [(0, 0), (620, 0), (1240, 0)])]

    # Convert white pixels to alpha
    word = white2alpha(word)

    # Here we save the word in current dataset
    word.save(dataset + f'word_{i}.png')

# Create & Store the high-low frequency deviants
hfd_word = Image.new('RGBA', wrd_size)    
lfd_word = Image.new('RGBA', wrd_size)    

hfd_chars = chars[chr_perm[hfd_idx]]
lfd_chars = chars[chr_perm[lfd_idx]]

[hfd_word.paste (chr_img, at) for chr_img, at in zip(hfd_chars, [(0, 0), (620, 0), (1240, 0)])]
[lfd_word.paste (chr_img, at) for chr_img, at in zip(lfd_chars, [(0, 0), (620, 0), (1240, 0)])]

# Convert white pixels to alpha
hfd_word = white2alpha (hfd_word)
lfd_word = white2alpha (lfd_word)

hfd_word.save(dataset + f'hfd_word.png')
lfd_word.save(dataset + f'lfd_word.png')



