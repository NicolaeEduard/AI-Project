import numpy as np
import os
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

#citesc label-urile de antrenament din train_labels si le stochez in vectorul train_labels
train_labels = np.genfromtxt('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', delimiter=',', dtype=int, skip_header=1, usecols=1)

#citesc label-urile de validare din validation_labels si le stochez in vectorul val_labels
val_labels = np.genfromtxt('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', delimiter=',', skip_header=1, usecols=1, dtype=int)

# atribui variabilei directory calea fisierului unde sunt stocate imaginile ce trebuie citite
directory = "/kaggle/input/unibuc-brain-ad/data/data"

# toate imaginile sunt adaugate intr-o lista, iar respectiva lista va fi sortata in ordine lexicografica
image_files = os.listdir(directory)
image_files = sorted(image_files)

#am facut o functie pentru a citi si a procesa fiecare imagine in parte
def read_image(file_path):
    img = Image.open(file_path).convert('L')  # fiecare imagine va fi convertita in grayscale
    img = img.resize((224, 224))              # apoi va fi adusa la dimensiunea ceruta, adica 224x224 pixeli
    img_array = np.array(img)                 # imaginea va fi transformată intr-un array
    img_array = img_array / 255.0             # se normalizeaza datele in valori de 0 si 1
    img.close()                               # pentru a elibera memoria RAM si a putea trece la urmatoarea imagine,
                                              # imaginea curenta este inchisa, ea fiind acum disponibila ca si vector
    return img_array

#am definit o functie generator pentru a putea accesa imaginile cate una pe rand,
# pentru o eficientizare a memoriei ocupate de program; aici ne vom folosi de functia definita si descrisa mai sus
def image_generator(image_files, directory):
    for imagine in image_files:
        file_path = os.path.join(directory, imagine)
        yield read_image(file_path)

# imaginile vor fi puse intr-un vector folosindu-ne de functia generator descrisa mai sus
all_images = np.array(list(image_generator(image_files, directory)))

# imaginile vor fi impartite in cele 3 sectiuni necesare,
#primele 15 mii fiind imagini de antrenare, urmatoarele 2 mii sunt imaginile de validare,
#iar ultimele 5149 vor fi imaginile necesare pentru crearea fisierului de submisie
train_images = all_images[:15000]
val_images = all_images[15000:17000]
submission_images = all_images[17000:]

# imaginile vor fi transformate in cate un tablou bidimensional (matrice)
train_images = train_images.reshape(train_images.shape[0], -1)
val_images = val_images.reshape(val_images.shape[0], -1)
submission_images = submission_images.reshape(submission_images.shape[0], -1)

# imaginile vor fi normalizate
scaler = StandardScaler()
scaler.fit(train_images)
train_images_scaled = scaler.transform(train_images)
val_images_scaled = scaler.transform(val_images)
submission_images_scaled = scaler.transform(submission_images)

# antrenam modelul Naive Bayes
model = GaussianNB()

batch_size = 1000
for i in range(0, len(train_images_scaled), batch_size):
    batch_images = train_images_scaled[i:i+batch_size]
    batch_labels = train_labels[i:i+batch_size]
    model.partial_fit(batch_images, batch_labels, classes=np.unique(train_labels))

# modelul va fi evaluat pe datele de validare
accuracy = model.score(val_images_scaled, val_labels)
print(accuracy)


# se va crea fisierul de output in formatul cerut in cerinta
submission_file = 'submission6.csv'
with open(submission_file, 'w') as f:
    f.write('id,class\n')
    k = 17001
    for i in range(submission_images_scaled.shape[0]):
        id_str = "{:06d}".format(i + k)  # coloana de id va fi formatata conform cerintei
        # ne folosim de modelul antrenat mai sus pentru a face predictii pe setul de date de submisie
        test_prediction = model.predict(submission_images_scaled[i].reshape(1, -1))[0]
        f.write('{},{}\n'.format(id_str, test_prediction))

print("Submission file created and saved as '{}'".format(submission_file))

