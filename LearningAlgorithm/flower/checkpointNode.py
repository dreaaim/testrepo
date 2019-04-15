from tensorflow.python import pywrap_tensorflow
import os
checkpoint_path = os.path.join( "drive/Flower/LearningAlgorithm/flower/model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:        
	print("node: ", key)
