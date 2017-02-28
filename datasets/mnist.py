# Test script for MNIST data visualization

import tensorflow as tf

print("Import and extract MNIST data...")
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist_data/", one_hot=True)

print("Create additional arrays...")
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

print("Implement model...")
y = tf.nn.softmax(tf.matmul(x, W) + b)

print("Implement cross-entropy...")
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

print("Launch TF session and initialize variables...")
tfsess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print("Init gradient descent algorithm...")
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

print("Run through training steps 1000 times...")
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    tfsess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print("Evalute predicted and correct data...")
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print()
print("Finished! Result (error %):")
print(tfsess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))