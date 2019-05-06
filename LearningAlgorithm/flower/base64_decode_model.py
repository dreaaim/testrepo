import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = "modelbase64"
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

sigs = {}

with tf.Graph().as_default() as g1:
	base64_str =tf.placeholder(tf.string, shape=[None],name='input_string')
	base64_scalar = tf.reshape(base64_str,[])
	input_str =tf.decode_base64(base64_scalar)
	decoded_image =tf.image.decode_png(input_str, channels=3)
	decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
							tf.float32)
	decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
	resize_shape = tf.stack([224,224])
	resize_shape_as_int = tf.cast(resize_shape, dtype =tf.int32)
	resize_image = tf.image.resize_bilinear(decoded_image_4d,
					resize_shape_as_int)
	print(resize_image.shape)
	tf.identity(resize_image, name="DecodeJPGOutput")

g1def = g1.as_graph_def()

with tf.Graph().as_default() as g2:
	with tf.Session(graph=g2) as sess:
		tf.saved_model.loader.load(sess, ["serve"], "./saved_model")
		graph = tf.get_default_graph()

		x = sess.graph.get_tensor_by_name("input:0")
		y = sess.graph.get_tensor_by_name("out:0")

g2def = g2.as_graph_def()

with tf.Graph().as_default() as g_combined:
	with tf.Session(graph=g_combined) as sess:
		x = tf.placeholder(tf.string ,name = "base64_input")
		y, = tf.import_graph_def(g1def, input_map={"input_string:0": x}, return_elements=["DecodeJPGOutput:0"])
		z, = tf.import_graph_def(g2def, input_map={"input:0": y,'keep_prob:0':1.0}, return_elements=["out:0"])
		tf.identity(z, "out")

		sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
			tf.saved_model.signature_def_utils.predict_signature_def(
				{"in": x}, {"out": z})

		builder.add_meta_graph_and_variables(sess,
						[tag_constants.SERVING],
						signature_def_map=sigs)

builder.save()