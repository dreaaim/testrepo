import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

filename_queue  = tf.train.string_input_producer(["andrive/Flower/LearningAlgorithm/flower/flower_train.tfrecords"])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
			features={
				'label': tf.FixedLenFeature([], tf.int64),
				'img_raw': tf.FixedLenFeature([], tf.string),})
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image,[224,224,3])
label = tf.cast(features['label'],tf.int32)
image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord,sess=sess)
	for i in range(20):
		example,l = sess.run([image,label])
		img=Image.fromarray(example,'RGB')
		print(img,l)
	coord.request_stop()
	coord.join(threads)