
"""
Data processing
"""
import io
import os
import lmdb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_image(val):

    img = Image.open(io.BytesIO(val))
    rp =64 / min(img.size)
    #img = img.resize(np.rint(rp * np.array(img.size)).astype(np.float32), Image.BICUBIC)
    img = img.resize(rp * np.array(img.size).astype(np.float32), Image.BICUBIC)
    img = img.crop((0,0,64,64))
    img = np.array(img, dtype=np.float32) / 255.
    return img

def iterate_images(DB_PATH,start_idx=None):
    
    with lmdb.open(DB_PATH, map_size=1099511627776,
                    max_readers=100, readonly=True) as env:
        with env.begin(write=False) as txn:
            with txn.cursor() as cursor:
                for i, (key, val) in enumerate(cursor):
                    if start_idx is None or start_idx <= i:
                        yield i, load_image(val)

def batched_images(batch_size, DB_PATH,start_idx=None) :

    batch, next_idx = None, None
    for idx, image in iterate_images(DB_PATH,start_idx):
        if batch is None:
            batch = np.empty((batch_size, 64, 64, 3))
            next_idx = 0
        batch[next_idx] = image
        next_idx += 1
        if next_idx == batch_size:
            yield idx + 1, batch
            batch = None
            
def plot(samples,img_size):
    
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(8,16)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_size, img_size,3))

    return fig

def view_samples(epoch, samples, nrows, ncols, figsize=(5,5)):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, 
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.axis('off')
        img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box-forced')
        im = ax.imshow(img, aspect='equal')
   
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes