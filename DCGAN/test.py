import tensorflow as tf  
import numpy as np
import os
import dcgan
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from IPython.display import clear_output
import argparse
from data import *
"""
This file tests the pre-trained model.

GAN model makes fake images.
* get_generator : This funciton calls pre-trained model's variables
* generate_samples : This function runs the model and generate fake images

"""

filePath = os.path.abspath("saved_models")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

args = argparse.ArgumentParser()

# Misc Parameters
args.add_argument("--allow_soft_placement", type=bool, default=False)
args.add_argument("--log_device_placement", type=bool, default=True)
args.add_argument("--allow_growth", type=bool, default=True)

# Test Parameters
args.add_argument("--checkpoint_dir", type=str, default=filePath)
args.add_argument("--num_classes", type=int, default=1)
args.add_argument("--batch_size", type=int, default=16)
args.add_argument("--z_dim", type=int, default=100)
args.add_argument("--img_size", type=int, default=64)


config, unparsed = args.parse_known_args()

checkpoint_file = tf.train.latest_checkpoint(config.checkpoint_dir)


def get_generator(batch_size,z_dim, num_classes):
    z = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim])
    Gan = dcgan.gan(num_classes)

    train = Gan.generate_sample(z,3,False )
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    return z, train

def generate_samples(checkpoint_dir,batch_size,z_dim, num_classes,img_size):
    with tf.device('/gpu:1'):
        z, train = get_generator(batch_size,z_dim, num_classes)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    if checkpoint_dir != None:
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            batch_z = np.random.normal(0, 1.0, [batch_size, z_dim]).astype(np.float32)
            rs = train.eval(feed_dict={z:batch_z})
    fig = plot(rs, img_size)
    plt.savefig('generated_sample.png',bbox_inchex='tight')
    plt.close(fig)
"""
    overall = []
    for i in range(4):
        temp = []
        for j in range(4):
            temp.append(rs[i * 4 + j])

        overall.append(np.concatenate(temp, axis=1))
    table = np.concatenate(overall, axis=0)

    plt.figure(figsize=[4, 4])
    plt.imshow(table)
    plt.savefig('./generated_sample.png')
    plt.show()
"""    
generate_samples(config.checkpoint_dir, config.batch_size, config.z_dim, config.num_classes, config.img_size)  
