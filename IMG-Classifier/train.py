import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# importa componentele principale din keras
#      - datasets: pt incarcarea setului CIFAR-10
#      - layers: pentru definirea arhitecturii retelei neuronale
#      - models: pentru a crea modelul secvential




# fct pt incarcare + preprocesarea datelor de antrenat si testat
def load_and_preprocess_data():
    # incarca setul de date cifar 10, 60000 32x32 colour images in 10 classes
    # care contine imagini si etichte de antrenat si testat
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # normalizeaza pixelii intre 0 si 1 -> imbunatateste antrenarea
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # am folosit doar o parte din setul complet pt eficienta
    train_images, train_labels = train_images[:30000], train_labels[:30000]
    test_images, test_labels = test_images[:5000], test_labels[:5000]

    # returneaza datele preprocesate pt antrenare si testarea
    return train_images, train_labels, test_images, test_labels




# construirea modelului CNN (convolutional neural network)
# este model sevential -> in care starurile sunt adaugate in ordine
def build_model():
    model = models.Sequential([
        # strat convolutional: 32 filtre de 3x3, functie de activare relu (Rectified Linear Unit)
        layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)), # dimensiunea unei imagini (32x32, RGB)

        # reduce dimensiunea spatiala , pooling 2x2, ajuta la generalizare
        layers.MaxPooling2D(pool_size=2),

        # adauga doua straturi convolutionale suplimentare cu 64 de filtre
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, activation='relu'),

        # aplatizeaza iesirea 3d intr-un vector 1d pt staturile dense
        layers.Flatten(),

        # strat complet cu 64 neuroni
        layers.Dense(64, activation='relu'),

        # strat de iesire, 10 neuroni (pt fiecare clasa),softmax produce probabilitati
        layers.Dense(10, activation='softmax')
    ])

    # compileaza modelul
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# antrenarea si salvarea modelului in format .h5
def model_trainer():
    # incarca si pregateste datele, preproc
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

    # creeaza modelul
    model = build_model()

    # antreneaza pt 10 epoci
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # afiseaza rezultatele
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    model.save('img_classificator.h5')  # salvaț ca .h5, compatibil cu Keras 3
    print("Model salvat cu succes!")

# daca e rulat se antreneaza modelul
if __name__ == "__main__":
    model_trainer()
