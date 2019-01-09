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

import numpy as np
import matplotlib.pyplot as plt

def GenerateFractal(image_name=None):

    if image_name is None:
        seed = 10 #np.random.randint(1000) #611 #10 #47 #88
        np.random.seed(seed)
        image_name = 'fractal-' + str(seed) + '.png'

    # Sizing parameters
    # n_dimensions controls dimension of the space
    # n_maxIterations controls how many points are generated
    # n_basisPoints controls how many basis points there are
    # n_grid controls the resolution of the plot
    n_dimensions = 2
    n_Iterations = int(4e6) #int(4e6)
    n_basisPoints = np.random.randint(3,8)
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

    # Initial point: can be any point in unit hypercube,
    # but first basis point works fine
    point = basis[0]

    # Create matrix to hold the density data
    density = np.zeros((n_grid,n_grid))
    
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

    # sparse version of density matrix is useful
    (rows,cols) = np.nonzero(density)
    vals = density[(rows,cols)]

    # Plot
    # Could do a simple imshow; all this is to get a particular look

    # DPI doesn't really seem to matter,
    # but getting the right marker size to avoid aliasing maybe takes some effort
    DPI = 100

    fig_dim = 2000.0/DPI
    markershape = 's'
    markersize = (3.1*72.0/DPI)**2 # 3 pixels wide? (markersize in square points)
    alphaval = 1.0
    min_shift = 1e-2

#    alphaval = 0.4
#    min_shift = 1e-2
#    # diamond
#    markershape = 'D'
#    fig_dim = 2000.0/DPI
#    markersize = 6
#    # "pixel" marker (basically square)
#    markershape = ','
#    fig_dim = 1700.0/DPI
#    markersize = 8.1
#    # square
#    markershape = 's'
#    fig_dim = 1600/DPI # pad_inches = fig_dim/4
#    markersize = 7.4

    # Map density values thru log
    # (log of density seems to produce more interesting image);
    # set minimum value for setting colormap
    logvals = np.log(vals)
    minv    = np.log(min_shift)

    # order the points so that darker points (higher values) are plotted last and on top
    ordering = np.argsort(logvals)
    rows = rows[ordering]
    cols = cols[ordering]
    logvals = logvals[ordering]

    plt.figure(figsize=(fig_dim,fig_dim), dpi=DPI, frameon=False)
    plt.scatter(cols,rows, c=logvals, s=markersize,marker=markershape,linewidths=0,\
                    cmap='Greys',norm=None,vmin=minv,alpha=alphaval)
    plt.axis([0,n_grid,0,n_grid], 'equal')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(image_name, bbox_inches='tight', pad_inches=fig_dim/6)

#    # imshow
#    # Consistent, good to check against
#    log_density = np.log(density + min_shift)
#    plt.figure(figsize=(fig_dim,fig_dim),dpi=DPI, frameon=False)
#    plt.imshow(log_density, cmap='Greys',origin='lower',interpolation='hamming')
#    plt.xticks([])
#    plt.yticks([])
#    plt.axis('off')
#    plt.savefig(image_name, bbox_inches='tight', pad_inches=fig_dim/4)

#    # could also write gnuplot code to automate plotting
#    with open('density.dat','w') as f:
#        for i in range(len(vals)):
#            f.write('{}, {}, {}\n'.format(rows[i],cols[i],vals[i]))
#    with open('plotCode.gp','w') as f:
#        f.write('inputName = \'density.dat\'\n')
#        f.write('outputName = \''+image_name+'\'')
#    #call(['gnuplot','plotDensity.gp']) 


if __name__ == "__main__":
    GenerateFractal()
