
import tensorflow as tf

"""
This function is generator.
It returns fake image. 
"""

def generator(z, batch_size, z_dim, is_training=False, reuse = False):
    with tf.variable_scope('generator', reuse=reuse):
        
        #layer1
        layer = tf.contrib.layers.fully_connected(z, 256, activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm)
        #layer2
        layer = tf.contrib.layers.fully_connected(layer,512, activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm)
        #layer3
        layer = tf.contrib.layers.fully_connected(layer, 2048, activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm)
        #layer4
        layer = tf.contrib.layers.fully_connected(layer, 64*64*3, activation_fn=tf.sigmoid, normalizer_fn=tf.contrib.layers.batch_norm)
        #reshape
        layer = tf.reshape(layer, tf.stack([batch_size, 64, 64, 3]),name='image')

        return layer