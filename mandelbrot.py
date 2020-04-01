"""
29 June 2019
SM Harwood
"""
import random
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_zoom(image_name='mbrot_zoom.png', max_iter=100):
    """ Sample complex plane, look for an interesting spot to zoom in """
   
    # ZOOM
    level = np.random.randint(4,7)
    print(level)
    n_samples = int(np.sqrt(100))
    # these work fairly well
    best_point = -1.0 + (1/3.0)*1j
    best_point = -0.5 + 0.5*1j
    # TODO: adaptively zoom;
    # maybe zoom more aggressively when there is more of the set nearby
    # or maybe zoom in on points in the set?
    best_point = -1.0 + 0.0*1j
    zoom = 0.7
    zoom_factor = 2.0
    zoom_levels = 20
    for i in range(zoom_levels):
        print("Zoom: {} at {}".format(zoom,best_point))
        x_samples = best_point.real + zoom*(np.random.rand(n_samples)-0.5)
        y_samples = best_point.imag + zoom*(np.random.rand(n_samples)-0.5)
        xmesh, ymesh = np.meshgrid(x_samples, y_samples)
        samples = (xmesh + ymesh*1j).flatten()
        sample_values = mandelbrot_iteration_optimized(samples, max_iter)
        if      all(sample_values >= max_iter) or \
            not any(sample_values >= max_iter):
            # everything sampled is in mandelbrot set
            # or nothing sampled in set
            # Don't zoom in, because we could be way off
            continue
        in_set = samples[sample_values >= max_iter]
        not_in_set = samples[sample_values < max_iter]
        distances = distance_matrix(in_set.reshape(-1,1), 
                                    not_in_set.reshape(-1,1))
        # find a point in mandelbrot set 
        # thats close to points not in set
        best_index = np.argmin(np.mean(distances, axis=1))
        best_point = in_set[best_index]
        zoom /= zoom_factor
        if zoom < 10**(-level):
            break

    # number of image pixels (on a edge)
    print(best_point)
    n_grid = 500
    xs = np.linspace(best_point.real-zoom, best_point.real+zoom, n_grid)
    ys = np.linspace(best_point.imag-zoom, best_point.imag+zoom, n_grid)
    xmesh, ymesh = np.meshgrid(xs, ys)
    points = xmesh + ymesh*1j
    values = mandelbrot_iteration_optimized(points, (level-2)*max_iter)
    grid = np.log(values)
    plot(grid, image_name)
    return

def mandelbrot_iteration_optimized(points, max_iter):
    """ 
    Do the iteration, specific to mandelbrot 
    points: np.array with complex entries
    """
    output = np.zeros(points.shape)
    z = np.zeros(points.shape, np.complex128)
    for it in range(max_iter):
        notdone = np.less(z.real*z.real + z.imag*z.imag, 4.0)
        output[notdone] = it + 1
        z[notdone] = z[notdone]**2 + points[notdone]
    return output

def plot(density, imagename):
    """ Plot it """
    # Plotting parameters
    n_grid = density.shape[0]
    DPI = 100
    fig_dim = (2.0*n_grid)/DPI
    
    cms = ['BuPu', 'twilight_shifted', 'viridis', 'cividis', 'cool', \
            'plasma', 'plasma_r', 'inferno', 'inferno_r', 'spring_r']
    cm = random.choice(cms)

    # imshow
    # Consistent, good to check against
    plt.figure(figsize=(fig_dim,fig_dim),dpi=DPI)
    plt.imshow(density, cmap=cm, origin='lower', interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(imagename, bbox_inches='tight', pad_inches=0)
    plt.close()
    return

def distance_matrix(x,y):
    """ 
    To avoid requiring scipy, just copy essence of distance_matrix from source 
    Parameters
    x : (M, K) ndarray - Matrix of M vectors in K dimensions.
    y : (N, K) ndarray - Matrix of N vectors in K dimensions.
    Returns
    result : (M, N) ndarray - Matrix containing the distance from every vector 
        in `x` to every vector in `y`.
    """
    # don't forget the abs in order to handle complex vectors
    return (np.sum(np.abs(y[np.newaxis,:,:] - x[:,np.newaxis,:])**2, axis=-1))**0.5


if __name__ == "__main__":
    mandelbrot_zoom()
