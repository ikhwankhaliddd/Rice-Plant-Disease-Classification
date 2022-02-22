import requests
import numpy as np

labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']


data = {
    "url" : "https://babel.litbang.pertanian.go.id/images/stories/blast.jpg"
}


url = "http://localhost:8080/functions/function/invocations"

results = requests.post(url, json=data).json()


print('[PREDICTION RESULT]')
print('+--------------------------------+')

score =[]

for category in results :
    print('+ {} : {}'.format(category, results[category]))
    score.append(results[category])

best_category = np.argmax(score)
print('+--------------------------------+')
print('Therefore, the model predicts the input image as {}'.format(labels[best_category].upper()))
print()