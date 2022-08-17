import numpy as np

def save_image_bin(grid,string,extension='bin'):
    fid = open('{}.{}'.format(string,extension),'w')
    gridfile = np.transpose(grid, (2,1,0))
    gridfile=gridfile.flatten()
    gridfile = np.float32(gridfile)
    gridfile.tofile(fid)
    fid.close()

def read_image_bin(string,gridsize,extension='bin'):
    fid = open('{}.{}'.format(string,extension),'r')
    grid = np.fromfile(fid,'float32')
    grid = grid.reshape(gridsize)
    grid = np.transpose(grid, (2,1,0))
    fid.close()
    return grid