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
import matplotlib.pyplot as plt

def generate_chaos(image_name=None, n_iter=5e5, verbose=False):
    """ Play the chaos game / generate a fractal"""
    if image_name is None:
        # For reproducibility while testing
        seed = np.random.randint(1000) #10 47 88 161 611 870
        np.random.seed(seed)
        image_name = 'fractal-' + str(seed) + '.png'

    # Sizing parameters
    # n_dimensions controls dimension of the space
    # n_iter controls how many points are generated
    # n_basisPoints controls how many basis points there are
    # n_grid controls the resolution of the plot
    n_dimensions = 2
    n_iter = int(n_iter)
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

    if verbose:
        print("Basis: " + str(basis))
        print("Move Fractions: "+str(moveFrac))
        print("Probabilities: "+str(cumulProb))

    # Initial point: can be any point in unit hypercube,
    # but first basis point works fine
    point = basis[0]

    # Create matrix to hold the density data
    density = np.zeros((n_grid,n_grid))
    
    if verbose:
        print("Starting chaos game iteration")

    # Choose sequence of basis points according to the weighting prob
    # Could do this manually by inverse CDF transform, but this is faster
    basis_sequence_full = np.random.choice(np.arange(n_basisPoints), p=prob, size=n_iter)

    # Sequence filtering does some interesting stuff in certain situations...
    if True:
        basis_sequence = basis_sequence_full
    else:
        basis_sequence = [basis_sequence_full[i] \
                          for i in range(1,len(basis_sequence_full)) \
                          if basis_sequence_full[i] != basis_sequence_full[i-1] ]

    # Generate more points in loop
    for i in basis_sequence:
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

    # Plot
    if verbose:
        print("Plotting")
    plotter(density, image_name)
    return

def plotter(density, image_name):
    """ Plot in a specific way """

    # Colormaps where something other than white is min
    specialmaps1 = ['plasma', 'cool']
    specialmaps2 = ['spring', 'cool_r', 'spring_r']
    n_grid = density.shape[0]

    # sparse version of density matrix is useful;
    # also, map thru log
    (rows,cols) = np.nonzero(density)
    vals = np.log(density[(rows,cols)])

    # Order the points so that higher values are plotted last and on top
    ordering = np.argsort(vals)
    rows = rows[ordering]
    cols = cols[ordering]
    vals = vals[ordering]

    # Try to classify/guess what the fractal will be like: 
    # Fraction of plot area that fractal takes up
    area_frac = len(vals)/float(n_grid**2)
    standard = True
    vmin = 0.0
    if area_frac < 0.01:
        # Probably uninteresting. Punch it up!
        colormap = random.choice(specialmaps2)
        standard = False
    elif area_frac < 0.10:
        colormap = random.choice(specialmaps1)
    else:
        # Classic bot
        # use only about 2/3 of Greys colormap
        colormap = 'Greys'
        vmin = -0.5*vals[-1]

    # Plotting parameters
    # DPI and fig_dim don't matter much;
    #   they are defined so we always get a 2000x2000 pixel image
    # But there is an interaction between DPI and marker size
    #   and the right marker size is required to avoid aliasing
    # DPI : dots per inch
    # fig_dim : figure dimension in inches
    # markershape : Shape of marker in scatterplot. 's' = square
    # markersize : Marker size in square points
    # vmin : Value that minimum of colormap corresponds to
    DPI = 100
    fig_dim = 2000.0/DPI
    markersize = lambda mw : (mw*72.0/DPI)**2
    markershape = 's' if standard else 'o'
    marker_width = 3.1 if standard else 10.0 # pixels
    facecolor = 'w' if standard else '0.10' # grayscale

#    # Every once in a while, add a pop of color
#    if np.random.rand() < 0.25:
#        # These colormaps can use the full spectrum
#        colormap = random.choice(specialmaps)
#        min_frac = 0
#    else:
#        # idea here: if the scatterplot takes up a fair amount of the plot area,
#        # allow a finer color gradation
#        colormap = 'Greys'
#        min_frac = max(0, 0.70 - area_frac)
#    # set minimum value for setting colormap
#    # min_frac : controls minimum value for colormap of scatterplot
#    #   min_frac = 0 : full colormap spectrum is used
#    #   min_frac = 1 : half of colormap spectrum is used
#    minv = -min_frac*max(vals)

    plt.figure(figsize=(fig_dim,fig_dim), dpi=DPI)
    plt.axis([0,n_grid,0,n_grid], 'equal')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    if not standard:
        plt.scatter(cols, rows, c=vals,
            s=markersize(marker_width*1.8),
            marker=markershape,
            cmap=colormap,
            vmin=vmin,
            alpha=0.2,
            linewidths=0)
    plt.scatter(cols, rows, c=vals,
        s=markersize(marker_width),
        marker=markershape,
        cmap=colormap,
        vmin=vmin,
        alpha=1.0,
        linewidths=0)
    plt.savefig(image_name, bbox_inches='tight', pad_inches=fig_dim/6, 
        facecolor=facecolor)
    plt.close() # to avoid memory issues in a loop
    return


if __name__ == "__main__":
    generate_chaos(None, 5e5, True)
