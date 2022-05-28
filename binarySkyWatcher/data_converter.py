import numpy as np
import cv2
import os

basedir = "/Users/infopz/Not_iCloud/ds_bilanciato/test/"
data_save = "/Users/infopz/Not_iCloud/ds_bilanciato/test.npm"
label_save = "/Users/infopz/Not_iCloud/ds_bilanciato/test_label.npm"

imgs_path = []

# Prima creo un elenco di tutti i file e le corrispettive label
labels = os.listdir(basedir)
for l in labels:
    if l == "3": continue #FIXME
    if l.startswith("."): continue  # skippo file nascosti
    label_p = basedir + str(l) + "/"
    # per ogni cartella delle label guardo i file dentro
    imgs = os.listdir(label_p)
    imgs = [i for i in imgs if i.endswith(".JPG") or i.endswith(".jpg")]
    lab = 0 if int(l)<3 else 1
    imgs_path += [(label_p+i, lab) for i in imgs]  # Creo tante coppie (path_immagine, label)


numImages = len(imgs_path)
print(numImages)

x = np.ndarray(shape=(numImages, 3, 416, 416), dtype=np.float32, order='C')

y = np.ndarray(shape=(numImages), dtype=np.uint8, order='C')

for i, (image_path, label) in enumerate(imgs_path):
    print(i)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.moveaxis(img, 2, 0)
    x[i] = img
    y[i] = int(label)


# NORMALIZZAZIONE
#means = np.mean(x, axis=(0,2,3))
#stds = np.std(x, axis=(0,2,3))
means = [142.40192, 160.8775, 181.65091]
stds = [50.12136, 40.37624, 36.41211]

for i in range(3):
    print("Processo canale", i)
    x[:,i,:,:] = (x[:,i,:,:] - means[i])/stds[i]


print(' x shape:', x.shape)
print(' x data type:', x[0].dtype)
print(' y shape:', y.shape)
print(' y data type:', y[0].dtype)

# Salvo i dati
f = np.memmap(data_save, dtype='float32', mode='w+', shape=(numImages, 3, 416, 416))
f[:] = x[:]
f.flush()

# Salvo le label
l = np.memmap(label_save, dtype='uint8', mode='w+', shape=(numImages,))
l[:] = y[:]
l.flush()
