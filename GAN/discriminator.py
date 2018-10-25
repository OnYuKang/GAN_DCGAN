
import tensorflow as tf

"""
This function is discriminator.
It returns probability of input image. 
If the reuse flag is set to True, variable is shared.
"""

    
def discriminator(inputs, keep_prob, is_training=False, reuse = False ):
     with tf.variable_scope('discriminator', reuse=reuse) as scope :
        if reuse:
            scope.reuse_variables()
            
        layer = tf.contrib.layers.fully_connected(tf.reshape(inputs, [128, 64*64*3]), 512, activation_fn=tf.nn.leaky_relu)
        layer = tf.nn.dropout(layer, keep_prob)
        
        layer = tf.contrib.layers.fully_connected(layer, 256, activation_fn=tf.nn.leaky_relu)
        layer = tf.nn.dropout(layer, keep_prob)
        
        layer = tf.contrib.layers.fully_connected(layer, 128, activation_fn=tf.nn.leaky_relu)
        layer = tf.nn.dropout(layer, keep_prob)
        
        layer = tf.contrib.layers.fully_connected(layer, 64, activation_fn=tf.nn.leaky_relu)
        layer = tf.nn.dropout(layer, keep_prob)
        
        logit = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None)
        
        return logit
