import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = 'C:\software\work\FireObjectClassify\data\iamges\\'
classes = {'bucket', 'FireExtinguishers', 'hose', 'IntakeScreen', 'siamese', 'StraightStreamNozzle', 'wye'}
writer =tf.python_io.TFRecordWriter("FireObject_train.tfrecords")

for index, name in enumerate(classes):
	class_path =cwd + name +'\\'
	for img_name in os.listdir(class_path):
		img_path = class_path + img_name
		img_raw = tf.gfile.FastGFile(img_path,'rb').read()
		#img = img.resize((224,224))
		#img_raw = img.tobytes()
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
			"image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
			"image/format" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg']))
		}))
		writer.write(example.SerializeToString())
writer.close()