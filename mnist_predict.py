import json
import os
from sys import argv

import cv2  # pip install opencv-python
import numpy as np
import onnxruntime


def predict(file, model="mnist_relu.onnx"):
    # Image loading
    img = cv2.imread(file, 0)

    if img is None:
        raise Exception("Image introuvable")

    img = cv2.resize(img, dsize=(28, 28),
                     interpolation=cv2.INTER_AREA)
    img.resize((1, 1, 28, 28))

    # Image to readable input
    data = json.dumps({'data': img.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')

    # Inference
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: data})
    prediction = int(np.argmax(np.array(result).squeeze(), axis=0))
    return prediction


# Arg check
if len(argv) != 3 or 0 > int(argv[2]) > 9:
    raise Exception("On attend deux arguments :\n\t1 - un dossier ou une image\n\t2 - le résultat attendu")

final_outputs = []


# Scan and compute all images in folder
if os.path.isdir(argv[1]):
    for file in os.listdir(argv[1]):
        prediction = predict(os.path.join(argv[1], file))
        final_outputs.append(prediction)
    # Print output
    print("Prédictions ", final_outputs.count(int(argv[2]))/len(final_outputs))

elif os.path.isfile(argv[1]):
    final_outputs.append(predict(argv[1]))
    print("Prédiction : ", *final_outputs, "\nRésultat attendu : ", argv[2])

else:
    raise Exception("Le chemin donné ne mène à rien.")
