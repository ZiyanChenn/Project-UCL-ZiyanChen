import numpy as np
from save_Annihil import *
from BinFile import *

for num in range(500): #change the file number here
    fileNumber =num + 1
    string = 'anni_'+str(fileNumber)
    fileName = string+'.txt'
    grid = save_annihil(fileName)
    save_image_bin(grid,string)