import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#delcare weights between input level and hidden level
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
#declare weights between hidden level and output level
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#define a sample
x=tf.constant([[0.7,0.9]])
#define hidden output
a=tf.matmul(x,w1)
#define final output
y=tf.matmul(a,w2)
with tf.Session() as sess:
	#init w1,w2
	sess.run(w1.initializer)
	sess.run(w2.initializer)
	print(sess.run(y))
