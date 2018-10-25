
import tensorflow as tf
from data import *
import dcgan
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from IPython.display import clear_output
import argparse
"""
This file trains the DCGAN Model.
"""


"""
Data path
"""
FOLDER_PATH = os.path.abspath("../lsun")
FOLDER_PATH_WITH_SLASH = FOLDER_PATH + "/"
DB_PATH = os.path.join(FOLDER_PATH_WITH_SLASH, "church_outdoor_train_lmdb")  


"""
Parameters
    Misc Parameters
    * allow_soft_placement : TensorFlow automatically choose an existing and supported device to run the operations in case the specified one doesn't exist, 
    * log_device_placement : Finds the device to which the operation or tensor is assigned
"""
# Specify GPU number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

args = argparse.ArgumentParser()

# Misc Parameters
args.add_argument("--allow_soft_placement", type=bool, default=False)
args.add_argument("--log_device_placement", type=bool, default=True)
args.add_argument("--allow_growth", type=bool, default=True)

# Training Parameters
args.add_argument("--img_size", type=int, default= 64)
args.add_argument("--channel", type=int, default=3)
args.add_argument("--batch_size", type=int, default=128)
args.add_argument("--z_dim", type=int, default=100)
args.add_argument("--num_epochs", type=int, default=25)
args.add_argument("--num_classes", type=int, default=1)
args.add_argument("--dataset", type=str, default='LSUN')

config, unparsed = args.parse_known_args()


"""
Session 
"""

session_conf = tf.ConfigProto(
    allow_soft_placement=config.allow_soft_placement,
    log_device_placement=config.log_device_placement)
session_conf.gpu_options.allow_growth=config.allow_growth
sess = tf.Session(config=session_conf)


"""
Training
"""
# placeholder
X = tf.placeholder(tf.float32, shape=[None, config.img_size,config.img_size,config.channel], name='input_x')
Z = tf.placeholder(tf.float32, shape=[None, config.z_dim], name='input_z')
IS_TRAINING = tf.placeholder(tf.bool,  name = 'is_training')

Gan = dcgan.gan(config.num_classes)

sample = Gan.generate_sample(Z, 3, IS_TRAINING)
 
d_logit_real, d_logit_fake = Gan.inference(X, sample, IS_TRAINING)
d_loss, g_loss = Gan.loss(d_logit_real, d_logit_fake)


# Get weights and bias to update
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
g_vars = [var for var in t_vars if var.name.startswith('generator')]


with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1=0.5, name='d_optimizer').minimize(d_loss,var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1=0.5, name='g_optimizer').minimize(g_loss,var_list=g_vars)
"""
Summary
"""
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", config.dataset))
d_loss_summary = tf.summary.scalar("dis_loss", d_loss)
g_loss_summary = tf.summary.scalar("gen_loss", g_loss)

train_summary_writer = tf.summary.FileWriter('./summary', sess.graph)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

step = 0 # for loss summary step
epoch_d_loss = []
epoch_g_loss = []

try:
    for epoch in range(0, config.num_epochs):
        batch_d_loss = []
        batch_g_loss = []
        for next_idx ,X_mb in batched_images(config.batch_size, DB_PATH, 1 ):            
            
            shape = X_mb.shape
            noise = np.random.uniform(-1., 1., size=[config.batch_size, config.z_dim])
                  
            _, d_summaries, d_loss_step = sess.run([d_optimizer, d_loss_summary, d_loss], feed_dict={X: X_mb, Z: noise, IS_TRAINING : True})
            _, g_summaries, g_loss_step = sess.run([g_optimizer, g_loss_summary, g_loss], feed_dict={X: X_mb, Z: noise, IS_TRAINING : True})
            
            clear_output(wait=True)
            
            print("Epoch %d: image %d " % (epoch, next_idx), flush=True)

            batch_d_loss.append(d_loss_step)
            batch_g_loss.append(g_loss_step)
            
            train_summary_writer.add_summary(d_summaries, step)
            train_summary_writer.add_summary(g_summaries, step)
            
            step = step + 1
        
        samples = sess.run(sample, feed_dict={Z:np.random.uniform(-1., 1., size=[4, config.z_dim]), IS_TRAINING : False})
        fig = plot(samples,config.img_size)
        
        try:
            if not (os.path.isdir("out")):
                os.makedirs(os.path.join('out'))
        except OSError  as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!")
                raise
            
            
        plt.savefig('out/test_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)    
        
        epoch_d_loss.append(np.mean(batch_d_loss))
        epoch_g_loss.append(np.mean(batch_g_loss))
        
        
        print('EPOCH: {}'.format(epoch))
        print('D loss: {:.4}'.format(np.mean(batch_d_loss)))
        print('G_loss: {:.4}'.format(np.mean(batch_g_loss)))
        print()
        
        saver.save(sess, "./saved_models/test%d_church.ckpt" % (epoch,))
    
except KeyboardInterrupt:
    print("Interrupted")
    
plt.plot(range(epoch+1),epoch_d_loss,'r--',label="discriminator")
plt.plot(range(epoch+1),epoch_g_loss,'bs:',label="generator")
plt.savefig('loss_graph.png',bbox_inches='tight')
plt.show()
