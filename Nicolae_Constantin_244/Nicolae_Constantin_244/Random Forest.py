import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# citim label-urile de antrenament si le stocam in vectorul train_labels
train_labels = pd.read_csv('/kaggle/input/unibuc-brain-ad/data/train_labels.txt')
# citim label-urile de validare si le stocam in vectorul validation_labels
validation_labels = pd.read_csv('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt')

# separam datele din coloana 'class' a datelor de antrenament si de validare in
#doi vectori separati
train_classes = train_labels['class'].values
validation_classes = validation_labels['class'].values

# citim label-urile de submisie si le stocam in vectorul sample_submission
sample_submission = pd.read_csv('/kaggle/input/unibuc-brain-ad/data/sample_submission.txt')

# separam datele din coloana 'class' a datelor de submisie intr-un vector separat
sample_ids = sample_submission['class'].values

# atribuim variabilei data_dir calea catre folderul unde sunt stocate imaginile
data_dir = '/kaggle/input/unibuc-brain-ad/data/data'

# atribuim variabilei image_size dimensiunea pe care trebuie sa o aiba imaginile
image_size = (224, 224)


# definim o functie pentru a citi si a procesa fiecare imagine individual
def preprocess_image(filename):
    # se selecteaza imaginea din folder
    image = Image.open(os.path.join(data_dir, filename))

    # imaginea este adusa la dimensiunea necesara, adica 224x224 pixeli
    image = image.resize(image_size)

    # imaginea este convertita la grayscale
    image = image.convert('L')

    # imaginea este adusa la forma unui tablou unidimensional
    image = np.array(image).reshape(-1)

    return image


# definim o functie pentru a procesa imaginile
def preprocess_images(filenames):
    # se creaza un vector pentru a stoca imaginile dupa ce acestea sunt procesate
    images = np.zeros((len(filenames), image_size[0] * image_size[1]))

    # parcurgem imaginile si le procesam pe fiecare in parte
    for i, filename in enumerate(filenames):
        images[i] = preprocess_image(filename)

    return images


# se creeaza si se sorteaza lista cu imagini extrase din data_dir
filenames = os.listdir(data_dir)
filenames = sorted(filenames)

# atribuim vectorului train_filenames primele 15 mii de imagini, cele necesare pentru antrenarea modelului
train_filenames = filenames[:15000]

# se preoceseaza imaginile de antrenament si acestea vor fi stocate in vectorul train_images
train_images = preprocess_images(train_filenames)

# se atribuie vectorului val_filenames cele 2 mii de imagini de validare
val_filenames = filenames[15000:17000]

# se proceseaza cele 2 mii de imagini de validare si acestea vor fi stocate in vectorul val_images
val_images = preprocess_images(val_filenames)

# vom antrena un model Random Forest pe datele de test
clf = RandomForestClassifier(class_weight="balanced", min_weight_fraction_leaf = 0.007, random_state=52)
clf.fit(train_images, train_classes)

# folosim modelul antrenat mai sus pentru a prezice tipul imaginilor de validare(daca apartin
# clasei 0 sau clasei 1)
val_pred = clf.predict(val_images)

# calculam acuratetea modelului Random Forest pe datele de validare si o afisam
val_acc = np.mean(val_pred == validation_classes)
print('Validation accuracy:', val_acc)

# se atribuie vectorului test_filenames cele 5149 de imagini de submisie
test_filenames = filenames[17000:]

# se proceseaza imaginile de submisie
test_images = preprocess_images(test_filenames)

# folosim modelul antrenat mai sus pentru a prezice tipul imaginilor de submisie(daca apartin
# clasei 0 sau clasei 1)
test_predictions = clf.predict(test_images)

# se creeaza fisierul de submisie in formatul cerut in cerinta
submission_file = 'submission2352352.csv'
with open(submission_file, 'w') as f:
    f.write('id,class\n')
    k = 17001
    for i, filename in enumerate(test_filenames):
        # se formateaza coloana 'id' sub forma ceruta in cerinta
        id_str = "{:06d}".format(i + k)
        # pentru fiecare id se va scrie clasa de care apartine prezisa de catre modelul
        #Random Forest
        class_label = test_predictions[i]
        f.write('{},{}\n'.format(id_str, class_label))

print("Submission file created and saved as '{}'".format(submission_file))

