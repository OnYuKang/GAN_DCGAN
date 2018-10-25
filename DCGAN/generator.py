
import tensorflow as tf

"""
This function is generator.
It returns fake image. 
"""

def generator(inputs, output_dim, alpha = 0.2, is_training=False, reuse = False):
    
    with tf.variable_scope('generator', reuse=reuse):
        layer1 = tf.layers.dense(inputs,4*4*1024, name='fnn')
        # reshape and the first layer
        layer1 = tf.reshape(layer1,(-1,4,4,1024))
        layer1 = tf.layers.batch_normalization(layer1,training=is_training)
        layer1 = tf.maximum(alpha * layer1, layer1)
        
        # second layer
        layer2 = tf.layers.conv2d_transpose(layer1,512, 5,strides=2,padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=is_training)
        layer2 = tf.maximum(alpha * layer2, layer2)
        
        # thrid layer
        layer3 = tf.layers.conv2d_transpose(layer2, 256, 5, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3,training=is_training)
        layer3 = tf.maximum(alpha * layer3, layer3)
        
        # fourth layer
        layer4 = tf.layers.conv2d_transpose(layer3, 128, 5, strides=2, padding='same')
        layer4 = tf.layers.batch_normalization(layer4,training=is_training)
        layer4 = tf.maximum(alpha * layer4, layer4)
        
        logits = tf.layers.conv2d_transpose(layer4, output_dim, 5, strides=2, padding='same')

        out = tf.sigmoid(logits)
        return out
        
        
