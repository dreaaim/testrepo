import os
import tensorflow as tf
from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np

def read_and_decode(filename):
	filename_queue =tf.train.string_input_producer([filename])
	reader =tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
				features={
					'label': tf.FixedLenFeature([], tf.int64),
					'img_raw': tf.FixedLenFeature([], tf.string),})
	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img = tf.reshape(img,[224,224,3])
	img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
	label =tf.cast(features['label'], tf.int64)
	return img, label

def create_batch(filename, batchsize):
	images, labels = read_and_decode(filename)
	min_after_dequeue = 10
	capacity = min_after_dequeue + 3 * batchsize

	image_batch, label_batch = tf.train.shuffle_batch([images, labels],
						batch_size = batchsize,
						capacity = capacity,
						min_after_dequeue = min_after_dequeue
						)

	label_batch =tf.one_hot(label_batch, depth=17)
	return image_batch, label_batch

def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict = { xs : v_xs, keep_prob : 1})
	correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict = { xs : v_xs, ys : v_ys, keep_prob : 1})
	return result

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1, dtype = tf.float32)
	return tf.Variable(initial, dtype = tf.float32, name = 'weight')

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape, dtype = tf.float32)
	return tf.Variable(initial, dtype = tf.float32, name = 'biases')

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides =[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

images_batch, labels_batch = create_batch("drive/Flower/LearningAlgorithm/flower/flower_train.tfrecords",40)

xs = tf.placeholder(tf.float32,[None,224,224,3],name="input")
ys = tf.placeholder(tf.float32,[None,17])
keep_prob = tf.placeholder(tf.float32,name="keep_prob")

xs = tf.reshape(xs,[-1,224,224,3])

W_conv1 = weight_variable([3,3,3,8])
b_conv1 = bias_variable([8])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3,3,8,16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([56*56*16, 32])
b_fc1 = bias_variable([32])
h_pool2_flat = tf.reshape(h_pool2, [-1,56*56*16])
h_fc1 =tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([32,17])
b_fc2 = bias_variable([17])

out = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name = "out")
prediction = tf.nn.softmax(out, name = "prediction")

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver(max_to_keep=1)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	accuracy = 0

	for i in range(1700):
		image_batch,label_batch = sess.run([images_batch, labels_batch])
		sess.run(train_step, feed_dict = {xs : image_batch, ys : label_batch, keep_prob : 0.9})
		accuracy = accuracy + compute_accuracy(image_batch, label_batch)
		if i%100 ==0:
			print(accuracy/100)	
			accuracy = 0	

	save_path = saver.save(sess,"aadrive/Flower/LearningAlgorithm/flower/model.ckpt")
	coord.request_stop()
	coord.join(threads)
		