from flask import Flask, render_template,request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re


import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load import * 

app = Flask(__name__)
global model, graph
model, graph = init()

	

def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))
	

@app.route('/')
def index():
	#initModel()
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	imgData = request.get_data()
	#print(imgData)
	convertImage(imgData)
	print "debug"
	x = imread('output.png',mode='L')
	x = np.invert(x)
	x = imresize(x,(28,28))
	#imshow(x)
	x = x.reshape(1,28,28,1)
	print "debug2"
	with graph.as_default():
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		print "debug3"
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
	#app.run(debug=True)
