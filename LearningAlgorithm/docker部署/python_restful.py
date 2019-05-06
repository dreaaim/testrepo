from __future__ import print_function

import base64
import requests
from urllib.request import Request
from urllib.request import urlopen
from urllib.parse import urlencode
import json

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://45.32.251.162:8501/v1/models/flower:predict'

# The image URL is the location of the image we should send to the server
image_url = 'https://ss0.bdstatic.com/94oJfD_bAAcT8t7mm9GUKT-xh_/timg?image&quality=100&size=b4000_4000&sec=1556013220&di=45185927263611b8ade8d2850fe73e8f&src=http://b-ssl.duitang.com/uploads/item/201501/20/20150120215958_dJwhf.thumb.700_0.jpeg'
image_path = 'C:\\software\\work\\LearningAlgorithm\\data\jpg\\bluebell\\image_0241.jpg'


def main():
	# Download the image
	dl_request = requests.get(image_url, stream=True)
	dl_request.raise_for_status()
	headers = {'content-type': 'application/x-www-form-urlencoded'}
	with open(image_path,'rb') as f:
		img = base64.b64encode(f.read()).decode()

	# Compose a JSON Predict request (send JPEG image in base64).
	#jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
	jpeg_bytes = str(base64.urlsafe_b64encode(dl_request.content),'utf-8')
	predict_request = {"instances" : [{"in": {"b64":jpeg_bytes}}]}
	json_data = {"instances" : [{"in": jpeg_bytes}]}
	#print(predict_request)
	predict_request = json.dumps(predict_request)
	#predict_request = bytes(urlencode(predict_request),encoding='utf8')
	# Send few requests to warm-up the model.
	for _ in range(3):
		a = requests.get("http://45.32.251.162:8501/v1/models/flower")
		print(a)
		response = requests.post(SERVER_URL, json=json_data,headers=headers)
		#req = urlopen(response)
		print(response.text)
		response.raise_for_status()

	# Send few actual requests and report average latency.
	total_time = 0
	num_requests = 10
	for _ in range(num_requests):
		response = requests.post(SERVER_URL, json=json_data,headers=headers)
		response.raise_for_status()
		total_time += response.elapsed.total_seconds()
		prediction = response.json()['predictions'][0]

	print('Prediction class: {}, avg latency: {} ms'.format(
		prediction['classes'], (total_time*1000)/num_requests))


if __name__ == '__main__':
	main()