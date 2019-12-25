"""
23 Dec 2018
SM Harwood

Generates points by an iterative bisection game:
    Given n "basis" points and an "active" point, choose a basis point at random.
    The new active point is a point a fraction of the distance 
    between the old active point and the chosen basis point

Related to Sierpinski triangle
Plots and saves to PNG
"""
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

def GenerateFractal(image_name=None,n_Iterations=4e6,verbose=False):

    if image_name is None:
        # For reproducibility while testing
        seed = np.random.randint(1000) #10 47 88 161 611 870
        np.random.seed(seed)
        image_name = 'fractal-' + str(seed) + '.png'

    # Sizing parameters
    # n_dimensions controls dimension of the space
    # n_Iterations controls how many points are generated
    # n_basisPoints controls how many basis points there are
    # n_grid controls the resolution of the plot
    n_dimensions = 2
    n_Iterations = int(n_Iterations)
    n_basisPoints = np.random.randint(3,7)
    n_grid = 1000
    
    # Set basis points of the game - RANDOM
    basis = np.random.rand(n_basisPoints,n_dimensions)

    # Shift and scale basis points to fill unit hypercube in at least one dimension
    # Calculate minimum and maximum (over all basis points) coordinate value 
    #   in each dimension
    # Shift for each dimension is minimum
    # Scale for each dimension is reciprocal of difference between max and min
    minCoordinate = np.min(basis, axis=0)
    maxCoordinate = np.max(basis, axis=0)
    scale = 1.0/np.abs(maxCoordinate - minCoordinate)
    # shift and scale each dimension of each basis point,
    # but clip to unit square (just to be sure)
    basis = np.minimum(1, scale*(basis - minCoordinate))

    # Set weighting for each point - RANDOM
    rawWeight = np.random.rand(n_basisPoints)
    sumWeight = np.sum(rawWeight)
    prob = rawWeight/sumWeight
    cumulProb = np.array([ np.sum(prob[0:i+1]) for i in range(n_basisPoints) ])

    # Set fraction of distance to each basis point to move - RANDOM
    moveFrac = np.random.rand(n_basisPoints)

#    print(basis)
#    print(moveFrac)
#    print(cumulProb)

    # Initial point: can be any point in unit hypercube,
    # but first basis point works fine
    point = basis[0]

    # Create matrix to hold the density data
    density = np.zeros((n_grid,n_grid))
    
    if verbose:
        print("Starting chaos game iteration")

    # Generate more points in loop
    for k in range(n_Iterations) :
        # Pick a basis point according to the weighting:
        #   generate random number uniformly in [0,1)
        #   find first index/basis point with cumulative probability
        #   greater than random uniform
        p = np.random.random()
        i = np.argmax(p < cumulProb)

        # Calculate point:
        #   some fraction of distance between previous point and basis point i
        point = (1-moveFrac[i])*point + moveFrac[i]*basis[i]

        # Copy to density matrix;
        # increment number of points occurring in the grid point of interest
        # Since the components of the points are all in the range [0,1],
        # floor the product of the point coordinate with the number of grid points
        # (and make sure that it's in the index range)
        x_coor = int(min(n_grid-1, np.floor(n_grid*point[0])))
        y_coor = int(min(n_grid-1, np.floor(n_grid*point[1])))
        density[y_coor][x_coor] += 1
    # end k loop

    if verbose:
        print("Done iterating")

    # sparse version of density matrix is useful
    (rows,cols) = np.nonzero(density)
    vals = density[(rows,cols)]

    # Plotting parameters
    # All this is to get a particular look with a scatterplot
    # DPI and fig_dim don't matter much;
    #   they are defined so we always get a 2000x2000 pixel image
    # But there is an interaction between DPI and marker size
    #   and the right marker size is required to avoid aliasing
    # DPI : dots per inch
    # fig_dim : figure dimension in inches
    # markershape : Shape of marker in scatterplot. 's' = square
    # markersize : Marker size in square points
    # alphaval : Alpha channel value for markers
    # min_frac : controls minimum value for colormap of scatterplot
    #   min_frac = 0 : full colormap spectrum is used
    #   min_frac = 1 : half of colormap spectrum is used
    DPI = 100
    fig_dim = 2000.0/DPI
    markershape = 's'
    markersize = (3.1*72.0/DPI)**2 # 3.1 pixels wide?
    alphaval = 1.0
    # white is the minimum value for these
    colormaps = ['YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn']
    # something other than white is min
    specialmaps = ['cool','plasma']

    # Every once in a while, add a pop of color
    if np.random.rand() < 0.25:
        # These colormaps can use the full spectrum
        colormap = random.choice(specialmaps)
        min_frac = 0
    else:
        # idea here: if the scatterplot takes up a fair amount of the plot area,
        # allow a finer color gradation
        colormap = 'Greys'
        min_frac = max(0, 0.70 - len(vals)/float(n_grid**2))

    # Map density values thru log
    # (log of density seems to produce more interesting image);
    # set minimum value for setting colormap
    logvals = np.log(vals)
    minv    = -min_frac*max(logvals)

    # order the points so that darker points (higher values) are plotted last and on top
    ordering = np.argsort(logvals)
    rows = rows[ordering]
    cols = cols[ordering]
    logvals = logvals[ordering]

    if verbose:
        print("Plotting")

    fig = plt.figure(figsize=(fig_dim,fig_dim), dpi=DPI)
    plt.scatter(cols,rows, c=logvals,s=markersize,marker=markershape,linewidths=0,\
                    cmap=colormap,norm=None,vmin=minv,alpha=alphaval)
    plt.axis([0,n_grid,0,n_grid], 'equal')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(image_name, bbox_inches='tight', pad_inches=fig_dim/6)
    plt.close() # to avoid memory issues in a loop

#    # imshow
#    # Consistent, good to check against
#    log_density = np.log(density + min_shift)
#    plt.figure(figsize=(fig_dim,fig_dim),dpi=DPI, frameon=False)
#    plt.imshow(log_density, cmap='Greys',origin='lower',interpolation='hamming')
#    plt.xticks([])
#    plt.yticks([])
#    plt.axis('off')
#    plt.savefig(image_name, bbox_inches='tight', pad_inches=fig_dim/4)


if __name__ == "__main__":
    GenerateFractal(None, 5e5)
