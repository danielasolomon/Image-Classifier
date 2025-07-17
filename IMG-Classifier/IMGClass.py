import cv2 as cv                                # pt citirea/redimensionarea imaginilor
import numpy as np
import matplotlib.pyplot as plt                 # afisare imagine
from tensorflow.keras.models import load_model  # incarcare model salvat

# lista claselor din cifar-10
CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#incarca si preproceseaza o imagine
def loaded_img(image_path):
    # citeste
    img = cv.imread(image_path)
    # converteste din bgr (default OpenCV) in rgb
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # redimensioneaza
    img = cv.resize(img, (32, 32))
    # normalizeaza
    img = img / 255.0
    return np.expand_dims(img, axis=0)


# fct pt a incarca modelul si pt a face predictie pe imaginea data
def predict_image(image_path, model_path='img_classificator.h5'):
    # incarca modelul salvat, preproceseaza imaginea si face o predictie
    model = load_model(model_path)
    img_array = loaded_img(image_path)
    prediction = model.predict(img_array)

    # gaseste clasa cu probabilitatea cea mai mare si eticheta asociata
    predicted_index = np.argmax(prediction)
    predicted_label = CLASS_NAMES[predicted_index]

    # afiseaza imaginea si predictia
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()

    print(f"Prediction: {predicted_label}")

if __name__ == "__main__":
    image_path = 'horse.jpg'
    predict_image(image_path)
