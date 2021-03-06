import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#use numpy to generate simulate data
from numpy.random import RandomState

#define batch size for trainning data
batch_size=8

#define parameters for neuro network
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#define input
x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#define neuro network processing
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#define lost function,
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#generate simulate datasets
rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)
Y=[[int(x1+x2<1)] for (x1,x2) in X]

#define session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print sess.run(w1)
	print sess.run(w2)
	STEPS=5000
	for i in range(STEPS):
		start=(i*batch_size)%dataset_size
		end=min(start+batch_size,dataset_size)
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i%1000==0:
			total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
			print("After %d training step(s),cross_entropy on all data is %g" %(i,total_cross_entropy))
	print sess.run(w1)
	print sess.run(w2)
