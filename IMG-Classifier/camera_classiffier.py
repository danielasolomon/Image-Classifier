import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# incarc modelul antrenat , definesc clasele
model = load_model('img_classificator.h5')
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# camera
cap = cv.VideoCapture(0)

# imaginea live, citeste frame-ul de la camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # redimensioneaza frame-ul la 32x32, normalizeaza si il face batch
    img_resized = cv.resize(frame, (32, 32)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # incarc imaginea in modelul meu sa faca predictia
    prediction = model.predict(img_resized)
    index = np.argmax(prediction)       # in functie de predictie itereaza in clasele mele si face corelatia cu eticheta
    label = class_names[index]

    # afiseaza text pe frame-ul captat
    cv.putText(frame, f'Prediction: {label}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # afiseaza fereastra video
    cv.imshow('Real-Time Classification', frame)

    # quit -> iese din camera
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
