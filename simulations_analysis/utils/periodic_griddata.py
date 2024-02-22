import numpy as np
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.spatial import cKDTree
from scipy.interpolate.ndgriddata import NearestNDInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

# periodic wrapper to griddata

def periodic_griddata(points, values, xi, period=None, method='linear',
             rescale=False):
    """
    Interpolate unstructured D-D data.

    Parameters
    ----------
    points : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
        Data point coordinates.
    values : ndarray of float or complex, shape (n,)
        Data values.
    xi : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
        Points at which to interpolate data.
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. One of

        ``nearest``
          return the value at the data point closest to
          the point of interpolation. See `NearestNDInterpolator` for
          more details.

        ``linear``
          tessellate the input point set to N-D
          simplices, and interpolate linearly on each simplex. See
          `LinearNDInterpolator` for more details.

        ``cubic`` (1-D)
          return the value determined from a cubic
          spline.

        ``cubic`` (2-D)
          return the value determined from a
          piecewise cubic, continuously differentiable (C1), and
          approximately curvature-minimizing polynomial surface. See
          `CloughTocher2DInterpolator` for more details.
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

        .. versionadded:: 0.14.0

    Returns
    -------
    ndarray
        Array of interpolated values.

    See Also
    --------
    LinearNDInterpolator :
        Piecewise linear interpolator in N dimensions.
    NearestNDInterpolator :
        Nearest-neighbor interpolator in N dimensions.
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    Notes
    -----

    .. versionadded:: 0.9

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------

    Suppose we want to interpolate the 2-D function

    >>> import numpy as np
    >>> def func(x, y):
    ...     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

    on a grid in [0, 1]x[0, 1]

    >>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

    but we only know its values at 1000 data points:

    >>> rng = np.random.default_rng()
    >>> points = rng.random((1000, 2))
    >>> values = func(points[:,0], points[:,1])

    This can be done with `griddata` -- below we try out all of the
    interpolation methods:

    >>> from scipy.interpolate import griddata
    >>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    >>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    >>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    One can see that the exact result is reproduced by all of the
    methods to some degree, but for this smooth function the piecewise
    cubic interpolant gives the best results:

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(221)
    >>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
    >>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)
    >>> plt.title('Original')
    >>> plt.subplot(222)
    >>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Nearest')
    >>> plt.subplot(223)
    >>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Linear')
    >>> plt.subplot(224)
    >>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Cubic')
    >>> plt.gcf().set_size_inches(6, 6)
    >>> plt.show()

    """ # numpy/numpydoc#87  # noqa: E501

    points = _ndim_coords_from_arrays(points)

    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    if ndim == 1:
        raise NotImplementedError("1-D periodic interpolation is currently not supported")

    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        from scipy.interpolate.interpolate import interp1d
        points = points.ravel()
        if isinstance(xi, tuple):
            if len(xi) != 1:
                raise ValueError("invalid number of dimensions in xi")
            xi, = xi
        # Sort points/values together, necessary as input for interp1d
        idx = np.argsort(points)
        points = points[idx]
        values = values[idx]
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False)
        return ip(xi)
    elif method == 'nearest':
        ip = NearestNDInterpolator(points, values, rescale=rescale)
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values,
                                  rescale=rescale)
    elif method == 'cubic' and ndim == 2:
        ip = CloughTocher2DInterpolator(points, values,
                                        rescale=rescale)
    else:
        raise ValueError("Unknown interpolation method %r for "
                         "%d dimensional data" % (method, ndim))
    if period is not None:
        ip.tree = cKDTree(points, boxsize=np.array(period)+0.001)
    return ip(xi)