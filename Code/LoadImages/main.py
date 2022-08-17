import numpy as np
from save_Annihil import *
from BinFile import *

for num in range(500):
    fileNumber =num + 1
    string = 'anni_'+str(fileNumber)
    fileName = string+'.txt'
    grid = save_annihil(fileName)
    save_image_bin(grid,string)

# coor = np.loadtxt("anni_1.txt")
# coor = coor.transpose()
#
# x = coor[0]
# y = coor[1]
# z = coor[2]
#
# dist = np.sqrt((x - 0.75)**2 + (y-0.75)**2 + (z-0.75)**2)
#
# # dist = np.linalg.norm(coor, axis=1)
# Rmean = np.mean(dist)
# print(Rmean)