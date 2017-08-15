import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load woeights into new model
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#loss,accuracy = model.evaluate(X_test,y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)
x = imread('output.png',mode='L')
x = np.invert(x)
x = imresize(x,(28,28))
# OSX errors from the following line
# imshow(x)
x = x.reshape(1,28,28,1)

out = loaded_model.predict(x)
print(out)
print(np.argmax(out,axis=1))
