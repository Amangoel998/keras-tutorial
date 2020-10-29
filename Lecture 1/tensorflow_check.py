import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# New change
tf.compat.v1.disable_eager_execution()

hello = tf.constant('Hello, Tensorflow version: '+str(tf.__version__))
sess = tf.compat.v1.Session()
print(sess.run(hello))