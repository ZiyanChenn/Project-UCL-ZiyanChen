import os

import numpy as np

from ActGate import *
from BinFile import *
from EllipGenerator import *
from RandomGenerate import *
from CreateHeader import *

# Based on Energy = 511 keV
# pixel = 2 mm
# size of lung box = 128 * 128 * 128 mm3
# 64 * 64 * 64
n = 64
size = [n,n,n]
width = [1.0,1.0,1.0]

path = './header'
os.chdir(path)

# Create 64x64x64 Lung Box:
# Lung Box
# Lung_Mu = 0.25 # (Max = 3.619, Min = 0)

#if want more files, just change the 500 to the number of files you want
for fileNumber in range(500): # 500

    # random (0.2-0.3)
    Box = np.ones((n, n, n)) * GenRandomLung()  # (0.2, 0.3) * 1000

    # Random Generator:
    times = GenRandomInt()

    # Generate Random Numbers of the Ellipsoids
    for t in range(times):

        # Location:
        Pos = ellip_Pos3d()
        # Orientation:
        Angle = ellip_Angle3d()
        # Scale:y
        Radius = ellip_Radius3d()
        # Mu
        Mu = GenRandomMu()

        Box = GenEllipsoid(Pos, Angle, Radius, Mu, Box, n)

    # Create .bin and .h33
    fileName = "AtnGate" + str(fileNumber+1)

    header = fileName+".h33"
    createH33Header(size,width,fileName+".bin",header)
    save_image_bin(Box.astype(np.float32),fileName)

# ActGate
ActGate = ActGate(n)
save_image_bin(ActGate,"ActGate")