import math
import numpy as np
import random

def split_tiles(M, n):
    """
    Takes a big tile M and separates it into a list of size nxn, containing tiles of equal size.
    For example:
    separate_tiles(M,2) would return the 4 quadrants of M; Q1,Q2,Q3,Q4 in the following format [[Q1,Q2],[Q3,Q4]]

    (Q1 is the north-western quadrant, Q2 north-easter, Q3 south-wester and Q4 south-easter)

    This fonction is approximately (because of trimming) such that:
    combine_tiles(n,separate_tiles(M,n)) == M
    """
    height, width = M.shape
    tile_h, tile_w = height//n, width//n
    tiles = np.zeros((n, n, tile_h, tile_w))

    for i in range(n):
        for j in range(n):
            tiles[i, j] = M[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
    return tiles
    


def combine_tiles(n, tiles):
    """ 
    Given a list of 2D tiles, combine them

    Parameters
    ----------
    """
    # check if inputs are correct
    if len(tiles) != n**2 and n>1:
        raise ValueError("The number of tiles should be the square of n. please check!")

    # check that input are the same size
    if not is_tiles_same_size(tiles):
        tiles = trim_tiles(tiles)


    # merged tiles together
    k, m = tiles[0].shape
    A = np.empty((n, n, k, m), dtype=tiles[0].dtype)
    for i in range(n):
        for j in range(n):
            tile_idx = i*n + j
            A[i, j, :, :] = tiles[tile_idx]

    return A


def trim_tiles(tiles): # TODO
    """ 
    Given a list of tiles that are not the same size, return a list of tiles
    that now have the same dimension. we start triming from the outside
    """
    min_height = min(tile.shape[0] for tile in tiles)
    min_width = min(tile.shape[1] for tile in tiles)
    
    trimed_tiles = []
    for tile in tiles:
        height_diff = tile.shape[0] - min_height
        width_diff = tile.shape[1] - min_width
        top = height_diff // 2
        bottom = height_diff - top
        left = width_diff // 2
        right = width_diff - left
        trimed_tiles.append(tile[top:tile.shape[0]-bottom, left:tile.shape[1]-right])
    
    return trimed_tiles


def is_tiles_same_size(tiles):
    """ 
    checking if all tiles are the same size

    Parameters
    ----------
    """
    tile_shape = tiles[0].shape
    is_same_size = all(tile.shape == tile_shape for tile in tiles)
    return is_same_size
    
#  -------------------------------------------------------------------------

def generate_shapes(n, x, y, epsilon):
    """ 
    ex: shapes = [(50, 70), (45, 68), (52, 75), (48, 72), (55, 80)]

    Parameters
    ----------
    n: int
        number of shape to generate
    x, y: int
        normal shape of arrays
    epsilon: int
        how many cells it can differ
    """
    shapes = []
    for _ in range(n):
        eps_x = random.randint(-epsilon, epsilon)
        eps_y = random.randint(-epsilon, epsilon)
        shapes.append((x+eps_x, y+eps_y))
    return shapes


#  -------------------------------------------------------------------------

def test2D_split_good():
    M = np.random.rand(36, 32)
    tiles = split_tiles(M, 2)
    print(tiles[0,1].shape)
    print(tiles[0,0].shape)
    print(tiles[1,0].shape)
    print(tiles[1,1].shape)
    
def test2D_split_bad():
    pass



def test2D_combine_good(): 
    n = 2
    tiles = [np.random.rand(36,36) for _ in range(n**2)]
    A = combine_tiles(n, tiles)
    print(A)

def test2D_combine_bad(): 
    n = 2
    epsilon = 4
    shapes = generate_shapes(n**2, 36, 36, 4)
    tiles = [ np.random.rand(shape[0], shape[1]) for shape in shapes ]
    A = combine_tiles(n, tiles)
    print(A)


def main():
    #  test2D_combine_good() # works!
    #  test2D_combine_bad() # works!
    test2D_split_good()
    
if __name__ == "__main__":
    main()
    

