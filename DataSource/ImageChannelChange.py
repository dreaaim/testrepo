import os
from PIL import Image

def change_image_channels(image, image_path):
	if image.mode == 'RGBA':
		r, g, b, a = image.split()
		image = Image.merge("RGB", (r, g, b))
		image.save(image_path)

	elif image.mode != 'RGB':
		image = image.convert("RGB")
		#os.remove(image_path)
		image.save(image_path)
	return image

if __name__ == "__main__":
	cwd = 'C:\software\work\FireObjectClassify\data\iamges\\'
	classes = {'bucket', 'FireExtinguishers', 'hose', 'IntakeScreen', 'siamese', 'StraightStreamNozzle', 'wye'}
	for index, name in enumerate(classes):
		class_path = cwd + name + '\\'
		for img_name in os.listdir(class_path):
			img_path = class_path +img_name
			img = Image.open(img_path)
			change_image_channels(img, img_path)