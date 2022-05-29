import os
import cv2
import numpy as np

base_dir = "/Users/infopz/Not_iCloud/ds_bilanciato/train/"

path_0 = []
for i in range(1,4):
    dir = base_dir+str(i)+"/"
    l = os.listdir(dir)
    l = [dir+i for i in l if i.endswith(".jpg")]
    path_0 += l

path_1 = []
for i in range(4,6):
    dir = base_dir + str(i) + "/"
    l = os.listdir(dir)
    l = [dir+i for i in l if i.endswith(".jpg")]
    path_1 += l

avg_0 = np.array([0.0, 0.0, 0.0])
avg_1 = np.array([0.0, 0.0, 0.0])

for i in path_0:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg = np.mean(img, axis=(0,1))
    avg_0 += avg

for i in path_1:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg = np.mean(img, axis=(0,1))
    avg_1 += avg

avg_0 = avg_0/len(path_0)
avg_1 = avg_1/len(path_1)

print(avg_0)
print(avg_1)