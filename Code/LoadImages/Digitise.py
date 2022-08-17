import numpy as np

def find_interval(elemx, elemy, elemz, xs, ys, zs):
    iterx = 0
    for x in xs:
        if (float(elemx) < x):
            break
        else:
            iterx += 1
    itery = 0
    for y in ys:
        if (float(elemy) < y):
            break
        else:
            itery += 1
    iterz = 0
    for z in zs:
        if (float(elemz) < z):
            break
        else:
            iterz += 1
    return [iterx, itery, iterz]

def resample(filecontent, sizex, sizey, sizez, spacing):
    # Compute lists of coordinate minima
    xs = np.arange(1, sizex + 1) * spacing - (sizex * spacing) / 2 * np.ones(sizex)
    ys = np.arange(1, sizey + 1) * spacing - (sizey * spacing) / 2 * np.ones(sizey)
    zs = np.arange(1, sizez + 1) * spacing - (sizez * spacing) / 2 * np.ones(sizez)
    grid1 = np.zeros([sizex, sizey, sizez])
    grid2 = np.zeros([sizex, sizey, sizez])
    for elem in filecontent:
        [x, y, z] = find_interval(elem[0], elem[1], elem[2], xs, ys, zs)
        try:
            grid1[x, y, z] += 1
        except IndexError:
            print(x, y, z, "out of bounds")
        [x2, y2, z2] = find_interval(elem[3], elem[4], elem[5], xs, ys, zs)
        try:
            grid2[x2, y2, z2] += 1
        except IndexError:
            print(x2, y2, z2, "out of bounds")
    return [grid1, grid2, xs, ys, zs]