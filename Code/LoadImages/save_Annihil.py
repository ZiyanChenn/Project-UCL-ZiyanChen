from Digitise import *
import numpy as npy

def save_annihil(fileName):

    spacingImage = 1.0 # mm

    [xdim,ydim,zdim] = [64, 64, 64]

    annihil = npy.loadtxt(fileName) # 1-500

    annihilList = annihil.tolist()
    xs = npy.arange(1, xdim + 1) * spacingImage - (xdim * spacingImage) / 2 * npy.ones(xdim)
    ys = npy.arange(1, ydim + 1) * spacingImage - (ydim * spacingImage) / 2 * npy.ones(ydim)
    zs = npy.arange(1, zdim + 1) * spacingImage - (zdim * spacingImage) / 2 * npy.ones(zdim)

    annihilationgrid = npy.zeros([xdim, ydim, zdim])

    for elem in annihilList:
        elem = npy.array(elem)
        [x, y, z] = find_interval(elem[0], elem[1], elem[2], xs, ys, zs)
        try:
            annihilationgrid[x, y, z] += 1
        except IndexError:
            print(x, y, z, "out of bounds")

    return annihilationgrid
