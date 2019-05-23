import tensorflow as tf
from PIL import Image

cwd = './CheckData/'
filename = tf.train.string_input_producer(['FireObject_train.tfrecords'])
reader = tf.TFRecordReader()
_, serializer = reader.read(filename)

feature = tf.parse_single_example(serializer,features={"label": tf.FixedLenFeature([], tf.int64),
					"img_raw": tf.FixedLenFeature([], tf.string)})

img = tf.decode_raw(feature['img_raw'], tf.uint8)
label = feature['label']
with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	for i in range(364):
		example, label_out = sess.run([img,label])
		img_new = sess.run(tf.reshape(example, [224,224,3]))
		img_show = Image.fromarray(img_new)
		img_show.save(cwd+str(i)+'_'+str(label_out)+'.jpg')
	coord.request_stop()
	coord.join(threads)
