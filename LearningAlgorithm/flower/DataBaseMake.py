import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = 'C:\software\work\LearningAlgorithm\data\jpg\\'
classes = {'daffodil', 'snowdrop', 'lilyvalley', 'bluebell', 'crocus', 'iris', 'tigerlily', 'tulip', 'fritiuary',
           'sunflower', 'daisy', 'coltsfoot', 'dandelion', 'cowslip', 'buttercup', 'windflower', 'pansy'}
writer =tf.python_io.TFRecordWriter("flower_train.tfrecords")

for index, name in enumerate(classes):
	class_path =cwd + name +'\\'
	for img_name in os.listdir(class_path):
		img_path = class_path + img_name
		img = Image.open(img_path)
		img = img.resize((224,224))
		img_raw = img.tobytes()
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
			"img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))
		writer.write(example.SerializeToString())
writer.close()