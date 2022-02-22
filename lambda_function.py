import tflite_runtime.interpreter as tflite
import numpy as np
from urllib.request import urlopen
from PIL import Image


# Create an interpreter interface for any model in TFLite
interpreter = tflite.Interpreter(model_path='model\padiq_classifier.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_index = input_details[0]['index']

output_details = interpreter.get_output_details()
output_index = output_details[0]['index']


def predict(X) :

    interpreter.set_tensor(input_index,X)
    interpreter.invoke()


    preds = interpreter.get_tensor(output_index)

    return preds[0]


labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']


def decode_prediction(pred) :
    result = {label: float(score) for label, score in zip(labels, pred)}
    return result

def preprocessor(img_url) :

    img = Image.open(urlopen(img_url))


    img = img.resize((150,150))


    X = np.expand_dims(img, axis = 0)

    
    X = X/255.0

    X = X.astype('float32')

    return X


def lambda_handler(event, context) :

    url = event['url']


    X = preprocessor(url)


    preds = predict(X)

    
    results = decode_prediction(preds)
    return results

event = {'url' : 'https://www.lampost.co/upload/penyakit-blas-serang-tanaman-padi-di-mesuji.jpg'}

results = lambda_handler(event, context = None)

print(results)