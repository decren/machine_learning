import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
a=tf.constant([1.0,2.0],name="a")
print a
b=tf.constant([2.0,3.0],name="b")
print b
result=a+b
print result
g=tf.Graph()
with g.device('/gpu:0'):
	with tf.Session() as sess:
		print(sess.run(result))
