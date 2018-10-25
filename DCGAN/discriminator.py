
import tensorflow as tf

"""
This function is discriminator.
It returns probability of input image. 
If the reuse flag is set to True, variable is shared.
"""

def discriminator(inputs, is_training, alpha =0.2, reuse = False):
    with tf.variable_scope('discriminator', reuse=reuse):
        
        # layer1
        layer1 = tf.layers.conv2d(inputs, 64, 5, strides=2, padding='same')
        layer1 = tf.maximum(alpha*layer1, layer1)
        
        # layer2
        layer2 = tf.layers.conv2d(layer1, 128, 5, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=is_training)
        layer2 = tf.maximum(alpha * layer2, layer2)
        
        # layer3
        layer3 = tf.layers.conv2d(layer2, 256, 5, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=is_training)
        layer3 = tf.maximum(alpha * layer3, layer3)
        
        # flatten
        flat = tf.reshape(layer3, (-1, 4*4*512))
        
        # layer4
        logits= tf.layers.dense(flat,1)
        out = tf.sigmoid(logits)

        return out, logits
    