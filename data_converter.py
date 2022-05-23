import numpy as np
import cv2
import os
import shutil

basedir = os.path.join(".","dataset", "train")
data_save = os.path.join(".","dataset", "trainF.npm")
label_save = os.path.join(".","dataset", "trainF_label.npm")

final_image_w = 416


imgs_path = []
# Prima creo un elenco di tutti i file e le corrispettive label
labels = os.listdir(basedir)

for l in labels:
    if l.startswith("."): continue  # skippo file nascosti che rompono
    label_p = os.path.join(basedir, l)
    # per ogni cartella delle label guardo i file dentro
    imgs = os.listdir(label_p)
    imgs = [i for i in imgs if i.endswith(".JPG") or i.endswith(".jpg")]
    imgs_path += [(os.path.join(label_p,i), l) for i in imgs] # Creo tante coppie (path_immagine, label)

numImages = len(imgs_path)
print(numImages)

x = np.ndarray(shape=(numImages, final_image_w, final_image_w, 3), dtype=np.uint8, order='C')

y = np.ndarray(shape=(numImages), dtype=np.uint8, order='C')

for i, (image_path, label) in enumerate(imgs_path):
    print(i)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x[i] = img
    y[i] = int(label) - 1 # -1 perche le cartelle vanno da 1 a 5, mentre torch vuole 0-4


# Li metto bene per il modello
x = np.moveaxis(x, 3, 1)
x = x.astype(np.float32)

print(' x shape:', x.shape)
print(' x data type:', x[0].dtype)
print(' y shape:', y.shape)
print(' y data type:', y[0].dtype)

#cancello eventuali dati precedenti
if os.path.exists(data_save):
    os.remove(data_save)

# Salvo i dati
f = np.memmap(data_save, dtype='float32', mode='w+', shape=(numImages, 3, final_image_w, final_image_w))
f[:] = x[:]
f.flush()

#cancello eventuali label precedenti
if os.path.exists(label_save):
    os.remove(label_save)

# Salvo le label
l = np.memmap(label_save, dtype='uint8', mode='w+', shape=(numImages,))
l[:] = y[:]
l.flush()

# test apertura
#fpr = np.memmap("/Users/infopz/Not_iCloud/train.npm", dtype='float32', mode='r', shape=(1352, 3, final_image_w, final_image_w))
#print(fpr[324])


''' HDF5 - VECCHIO METODO
import h5py
with h5py.File('train.hdf5', 'w') as hf:
    dset_x_train = hf.create_dataset('x_train', data=x, shape=(numImages, final_image_w, final_image_w, 3), compression='gzip', chunks=True)
    dset_y_train = hf.create_dataset('y_train', data=y, shape=(numImages,), compression='gzip', chunks=True)

#with h5py.File('test.hdf5', 'w') as hf:
#    dset_x_test = hf.create_dataset('x_test', data=x_img_test, shape=(10000, 32, 32, 3), compression='gzip', chunks=True)
#    dset_y_test = hf.create_dataset('y_test', data=y_label_test, shape=(10000, 1), compression='gzip', chunks=True)


with h5py.File('train.hdf5', 'r') as hf:
    dset_x_train = hf['x_train']
    dset_y_train = hf['y_train']

    print(dset_x_train)
    print(dset_y_train)
    
'''
