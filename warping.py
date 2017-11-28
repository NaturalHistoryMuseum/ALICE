from numba import njit
import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
import skimage
import skimage.io
from skimage.transform import estimate_transform


@njit
def coords_to_index(coords, shape):
    j, i = coords
    return i * shape[1] + j


@njit
def closest_corners(location, grid_dimensions, grid_separation):
    x, y = location[0] / grid_separation[1], location[1] / grid_separation[0]
    x1 = np.floor(x)
    y1 = np.floor(y)
    x2 = x1 + 1
    y2 = y1 + 1

    corners = []
    weights = []

    # UPPER LEFT
    corners.append(coords_to_index((x1, y1), grid_dimensions))
    weights.append((x2 - x) * (y2 - y))

    # UPPER RIGHT
    corners.append(coords_to_index((x2, y1), grid_dimensions))
    weights.append((x - x1) * (y2 - y))

    # LOWER RIGHT
    corners.append(coords_to_index((x2, y2), grid_dimensions))
    weights.append((x - x1) * (y - y1))

    # LOWER LEFT
    corners.append(coords_to_index((x1, y2), grid_dimensions))
    weights.append((x2 - x) * (y - y1))

    return corners, weights


@njit
def sparse_data_weights(P, grid_dimensions, grid_separation):
    num_features = P.shape[0]
    w = np.zeros(num_features * 8)
    row = np.zeros(num_features * 8)
    col = np.zeros(num_features * 8)

    index = 0
    for i in range(num_features):
        p = P[i]

        for vertex_index, weight in zip(*closest_corners(p, grid_dimensions, grid_separation)):
            for co in range(2):
                w[index] = weight
                row[index] = i * 2 + co
                col[index] = vertex_index * 2 + co
                index += 1

    return w, (row, col)


class SimilarAsPossible:
    '''Based on section 3.2 from 'Bundled Camera Paths for Video Stabilization'.'''
    R90 = coo_matrix(np.array([[0, 1], [-1, 0]]))

    def __init__(self, grid_shape, grid_separation):
        self.height, self.width = grid_shape
        self.num_vertices = (self.height + 1) * (self.width + 1)
        self.grid_separation = grid_separation
        self.s = grid_separation[0] / grid_separation[1]

    def data_term(self, P, P_hat):
        num_features = P.shape[0]
        A = coo_matrix(sparse_data_weights(P, (self.height + 1, self.width + 1), self.grid_separation),
                       shape=(num_features * 2, self.num_vertices * 2))
        b = P_hat.flatten()
        return A, b

    def triangle_indices(self, coords):
        corners = np.array(coords).T
        indices = np.ravel_multi_index(corners, (self.height + 1, self.width + 1))
        x_indices = indices * 2
        y_indices = indices * 2 + 1
        combined_indices = np.array(list(zip(x_indices, y_indices))).flatten()
        return coo_matrix((np.ones(6), (np.arange(6), combined_indices)), (6, self.num_vertices * 2))

    @property
    def triangles(self):
        for i in range(self.height):
            for j in range(self.width):
                # this is a quad -> two triangles
                yield self.triangle_indices([[i, j], [i + 1, j + 1], [i + 1, j]])
                yield self.triangle_indices([[i + 1, j + 1], [i, j], [i, j + 1]])

    def shape_preserving_term(self):
        c = coo_matrix(([1, 1], ((0, 1), (0, 1))), shape=(2, 6))
        c0 = coo_matrix(([1, 1], ((0, 1), (2, 3))), shape=(2, 6))
        c1 = coo_matrix(([1, 1], ((0, 1), (4, 5))), shape=(2, 6))

        A = vstack([(c - c1 - self.s * self.R90 * (c0 - c1)) * t for t in self.triangles])
        b = np.zeros(A.shape[0])

        return A, b

    def fit(self, P, P_hat, alpha):
        A_data, b_data = self.data_term(P, P_hat)
        A_shape, b_shape = self.shape_preserving_term()
        A = vstack([A_data, alpha**2 * A_shape])
        b = np.concatenate([b_data, alpha**2 * b_shape])

        x_points = np.arange(self.height + 1) * self.grid_separation[0]
        y_points = np.arange(self.width + 1) * self.grid_separation[1]
        X, Y = np.meshgrid(x_points, y_points)
        grid = np.stack((X.flatten(), Y.flatten()), axis=1)

        r = b - (A * grid.reshape(-1, 1)).flatten()

        self.V = lsqr(A, r, show=True)[0] + grid.flatten()

        self.transformation = estimate_transform('piecewise-affine', grid, self.V.reshape(-1, 2))


def bidirectional_similarity(u, v, u1, v1):
    height, width = u.shape
    Y, X = np.mgrid[:height, :width]

    y_projected = skimage.transform.warp(Y * 1.0, np.stack((Y - v, X - u)))
    x_projected = skimage.transform.warp(X * 1.0, np.stack((Y - v, X - u)))

    y_reprojected = skimage.transform.warp(y_projected, np.stack((Y - v1, X - u1)))
    x_reprojected = skimage.transform.warp(x_projected, np.stack((Y - v1, X - u1)))

    return (Y - y_reprojected)**2 + (X - x_reprojected)**2
