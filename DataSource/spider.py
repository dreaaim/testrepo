import re
import requests
import time

i = 114

def downloadPic(html, keyword):
	global i
	pic_url = re.findall('"objURL":"(.*?)"',html,re.S)
	print(pic_url)
	print('开始找'+keyword+'的图片')
	for each in pic_url:
		print('正在下载第'+str(i)+'张图片，图片地址：'+ str(each))
		try:
			pic = requests.get(each, timeout=10)
		except Exception as e:
			print('图片无法下载,错误原因：')
			print(e)
			continue

		dir = 'c:/software/work/spider/images/'+ keyword + '_' + str(i) + '.jpg'
		fp = open(dir, 'wb')
		fp.write(pic.content)
		fp.close()
		i+=1

if __name__ == '__main__':
	word = input("Input key word:")
	for epoch in range(3,20):
		num = epoch * 60
		url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word   + '&pn=' + str(num)
		result = requests.get(url)
		downloadPic(result.text, word)
		time.sleep(10)

