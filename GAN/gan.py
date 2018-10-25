
import tensorflow as tf
from generator import *
from discriminator import *

"""
This file merges generator file and discriminator file.

The gan class have three functions

* generate_sample : Using generator function, return sample image by generator
* inference : Using discriminator function, return logit of fake and real image
* loss : Compute loss of generator and discriminator
"""

class gan(object):
    def __init__(self, num_classes):
        super(gan, self).__init__()
        self.num_classes = num_classes

    def generate_sample(self,inputs,batch_size, z_dim, is_training):
        return generator(inputs, batch_size, z_dim, is_training=is_training, reuse = False)
    
    def inference(self, X, sample, IS_TRAINING):
        d_logit_real = discriminator(X,keep_prob=0.5, is_training=IS_TRAINING, reuse=False )
        d_logit_fake = discriminator(sample, keep_prob=0.5, is_training=IS_TRAINING, reuse=True)

        return d_logit_real, d_logit_fake

    def loss(self,d_logit_real,d_logit_fake,reuse=False):
        
        with tf.name_scope('loss'):
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)*0.9),name='D_real_cross_entropy_mean')
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)),name='D_fake_cross_entropy_mean')
            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)),name='G_cross_entropy_mean')

            return self.d_loss, self.g_loss