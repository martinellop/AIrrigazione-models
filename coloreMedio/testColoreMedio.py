import os
import cv2
import numpy as np

avg0 = np.array([121.5232, 145.4649, 172.6319])
avg1 = np.array([175.5577, 181.4360, 187.9086])

base_dir = "/Users/infopz/Not_iCloud/ds_bilanciato/test/"

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


correct = 0
for i in path_0:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg = np.mean(img, axis=(0, 1))
    diff0 = np.sum(np.abs(avg-avg0))
    diff1 = np.sum(np.abs(avg-avg1))
    if diff0 < diff1:
        correct += 1

acc0 = correct/len(path_0)

correct = 0
for i in path_1:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg = np.mean(img, axis=(0, 1))
    diff0 = np.sum(np.abs(avg-avg0))
    diff1 = np.sum(np.abs(avg-avg1))
    if diff1 < diff0:
        correct += 1

acc1 = correct/len(path_1)
print("Class 0 accuracy:", acc0)
print("Class 1 accuracy:", acc1)

totalAcc = 0.6 * acc0 + 0.4 * acc1
print("Total Accuracy:", totalAcc)