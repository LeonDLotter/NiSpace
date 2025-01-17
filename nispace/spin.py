import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from sklearn.manifold import MDS
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .nulls import find_vol_parc_centroids


def _distance_matrix(coords1, coords2):
    return np.linalg.norm(coords1[:, np.newaxis] - coords2, axis=2)


def coords_from_distmat(distmat, n_dim=3):
    """
    Constructs coordinates from a distance matrix using Multidimensional Scaling (MDS).

    Parameters:
    - distance_matrix: A square matrix where element (i, j) represents the distance between points i and j.
    - n_components: The number of dimensions for the output coordinates (e.g., 2 for 2D, 3 for 3D).

    Returns:
    - A numpy array of shape (n_samples, n_components) containing the constructed coordinates.
    """
    mds = MDS(n_components=n_dim, dissimilarity="precomputed", random_state=42)
    coordinates = mds.fit_transform(distmat)
    return coordinates


def largest_distance(coords, reference_coord):
    """
    Finds the largest Euclidean distance between a list of 3D coordinates and a single reference coordinate.

    Parameters:
    - coords: A list or array of 3D coordinates, where each coordinate is a tuple or list (x, y, z).
    - reference_coord: A single 3D coordinate (x, y, z) to compare against.

    Returns:
    - The largest distance found.
    """
    coords = np.array(coords)
    reference_coord = np.array(reference_coord)
    
    # Calculate the Euclidean distance from each point to the reference point
    distances = np.linalg.norm(coords - reference_coord, axis=1)
    
    # Find the maximum distance
    max_distance = np.max(distances)
    return max_distance


def create_sphere_surface(radius, num_points=100, center=(0, 0, 0)):
    """
    Creates a surface of a sphere around a given center and returns the coordinates of the sphere surface.

    Parameters:
    - center: A tuple or list of (x, y, z) coordinates for the center of the sphere.
    - radius: The radius of the sphere.
    - num_points: The number of divisions along each spherical coordinate. 

    Returns:
    - A numpy array of shape (num_points^2, 3) containing the 3D coordinates of the sphere surface points.
    """
    # Create a grid of angles
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]

    return np.vstack((x.ravel(), y.ravel(), z.ravel())).T


def assign_parcels_to_sphere(data_obs, sphere_points):
    """
    Assigns parcels to points on a sphere.

    Parameters:
    - data_obs: A numpy array of shape (n, 3) containing the 3D coordinates of the parcels.
    - sphere_points: A numpy array of shape (m, 3) containing the 3D coordinates of the sphere points.

    Returns:
    - A numpy array of shape (n,) containing the indices of the closest points on the sphere for each parcel.
    """
    tree = KDTree(sphere_points)
    _, indices = tree.query(data_obs)
    return indices


def rotate_sphere(sphere_coords, angle, axis):
    """
    Rotates a set of 3D points around a fixed axis.

    Parameters:
    - sphere_coords: A numpy array of shape (n, 3) containing the 3D coordinates of the points to rotate.
    - angle: The angle of rotation in radians.
    - axis: A tuple or list of (x, y, z) coordinates for the axis of rotation.

    Returns:
    - A numpy array of the rotated 3D coordinates.
    """
    rotation = Rotation.from_rotvec(angle * np.array(axis))
    return rotation.apply(sphere_coords)


def rotate_sphere_random(sphere_coords, center, random_state=None, axis=None, mirror_along_axis=None):
    """
    Rotates a set of 3D points randomly around a fixed center.

    Parameters:
    - sphere_coords: A numpy array of shape (n, 3) containing the 3D coordinates of the points to rotate.
    - center: A tuple or list of (x, y, z) coordinates for the center of the sphere.
    - random_state: An integer or numpy random state for reproducibility.
    - axis: The axis to rotate around: "x", "y", or "z".
    
    Returns:
    - A numpy array of the rotated 3D coordinates.
    """
    # Translate points to origin
    translated_points = sphere_coords - center
    
    # Random state
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
       
    # Get random rotation
    if axis is None:
        # Random rotation in any direction
        random_rotation = Rotation.random(random_state=random_state)
    else:
        # Random rotation around a fixed axis
        angle = random_state.uniform(0, 2 * np.pi)
        random_rotation = Rotation.from_euler(axis, angle)
            
    # Apply the random rotation to the translated points
    rotated_translated_points = random_rotation.apply(translated_points)
    
    # Translate points back to the original center
    rotated_points = rotated_translated_points + center
    
    if mirror_along_axis is not None:
        mirror_axis = ["x", "y", "z"].index(mirror_along_axis)
        
        # Convert rotation to rotation vector
        rotvec = random_rotation.as_rotvec()
        
        # Mirror the rotation vector along the specified axis
        # this is done by negating the two axes that are not the mirror axis
        rotvec = -rotvec
        rotvec[mirror_axis] = -rotvec[mirror_axis]
        
        # Create a new rotation from the mirrored rotation vector
        random_rotation_mirror = Rotation.from_rotvec(rotvec)
        
        # Apply the random rotation to the translated points
        rotated_translated_points_mirror = random_rotation_mirror.apply(translated_points)
        
        # Translate points back to the original center
        rotated_points_mirror = rotated_translated_points_mirror + center
        
        return rotated_points, rotated_points_mirror
    
    else:
        return rotated_points
    
    
def _k_nearest_neighbor_matrix(coords, k=100):
    """Construct a k-nearest neighbor matrix (graph)."""
    dist_mat = _distance_matrix(coords, coords)
    np.fill_diagonal(dist_mat, np.inf)
    neighbors = np.argsort(dist_mat, axis=1)[:, :k]
    graph = np.zeros_like(dist_mat)
    for i, nbrs in enumerate(neighbors):
        graph[i, nbrs] = 1
    return graph


def _inverse_distance_matrix(coords):
    """Construct an inverse distance matrix (graph)."""
    dist_mat = _distance_matrix(coords, coords)
    with np.errstate(divide='ignore'):
        inv_dist_mat = 1.0 / dist_mat
    inv_dist_mat = np.nan_to_num(inv_dist_mat, nan=0.0)
    np.fill_diagonal(inv_dist_mat, 0)  # No self-loops
    return inv_dist_mat


def _laplacian_matrix(graph):
    """Compute the graph Laplacian."""
    degree_matrix = np.diag(graph.sum(axis=1))
    laplacian = degree_matrix - graph
    return laplacian


def match_coord_sets(coords1, coords2, method="graph_matching", k=None):
    """
    Finds the best one-to-one matching between two lists of coordinates.

    Parameters:
    - coords1: A list or array of 3D coordinates, where each coordinate is a tuple or list (x, y, z).
    - coords2: A list or array of 3D coordinates, where each coordinate is a tuple or list (x, y, z).
    - method: The method to use for matching: "hungarian" or "pygmtools".
    - k: The number of nearest neighbors to use for the graph matching.

    Returns:
    - A list of tuples, where each tuple contains the indices of the matched coordinates from coords1 and coords2.
    """
    # Ensure the input lists are numpy arrays
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    if method == "graph_matching":
        
        # get graphs
        if k is None:
            g1, g2 = _k_nearest_neighbor_matrix(coords1), _k_nearest_neighbor_matrix(coords2)
        else:
            g1, g2 = _k_nearest_neighbor_matrix(coords1, k), _k_nearest_neighbor_matrix(coords2, k)
        
        # get laplacian
        laplacian1, laplacian2 = _laplacian_matrix(g1), _laplacian_matrix(g2)
        
        # get eigenvectors
        eigvec1, eigvec2 = np.linalg.eig(laplacian1)[1], np.linalg.eig(laplacian2)[1]
        
        # distance matrix between eigenvectors
        distmat = _distance_matrix(eigvec1, eigvec2)
        
    elif method == "hungarian":
        # Calculate the distance matrix
        distmat = _distance_matrix(coords1, coords2)

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distmat)

    # Return the list of matched indices
    return np.stack([row_ind, col_ind], axis=1)


def weighted_average_match(coords1, coords2, data, weight_power=1.0):
    """
    Calculates a weighted average of data based on the distance between original and rotated coordinates.

    Parameters:
    - coords1: A numpy array of shape (n, 3) containing the original 3D coordinates.
    - coords2: A numpy array of shape (n, 3) containing the rotated 3D coordinates.
    - data: A numpy array of shape (n,) containing the data values associated with coords1.
    - weight_power: The power to use for the distance weighting; 0 < weight_power < inf.

    Returns:
    - A numpy array of shape (n,) containing the weighted average of the data.
    """
    if coords1.shape != coords2.shape:
        raise ValueError(f"coords1 {coords1.shape} and coords2 {coords2.shape} must have the same shape")
    
    # Calculate the distance matrix
    distance_matrix = _distance_matrix(coords1, coords2)
    
    # Normalize distance matrix to be between 0 and 1
    #distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())
    
    with np.errstate(divide='ignore', invalid='ignore'):
        
        # Calculate weights as the inverse of distances raised to the power
        weights = np.power(1 / distance_matrix, weight_power)
        weights[~np.isfinite(weights)] = 0.0
        
        # Normalize weights
        weights /= weights.sum(axis=1, keepdims=True)
        weights[np.isnan(weights)] = 0.0

    # if a region has only zero weights, the closest coordinate is set to 1.0
    zero_weight_rows = np.all(weights == 0.0, axis=1)
    min_distance_indices = np.argmin(distance_matrix[zero_weight_rows], axis=1)
    weights[zero_weight_rows, min_distance_indices] = 1.0
    
    # Calculate the weighted average
    weighted_data = np.dot(weights, data)
    
    return weighted_data


def get_rotated_data(data, coords1, coords2, method, distance_power=None):
    """
    Rotates data based on the method and distance power.
    
    Parameters:
    - data: A numpy array, or a len(2) list of numpy arrays, of shape (n,) containing the data values. 
    - coords1: A numpy array, or a len(2) list of numpy arrays, of shape (n, 3) containing the original 3D coordinates.
    - coords2: A numpy array, or a len(2) list of numpy arrays, of shape (n, 3) containing the rotated 3D coordinates.
    - method: The method to use for rotation: "hungarian" or "distance_weighted".
    - distance_power: The power to use for the distance weighting.
    
    Returns:
    - A numpy array of shape (n,) containing the rotated data.
    """
    if isinstance(data, np.ndarray):
        data = [data]
    if isinstance(coords1, np.ndarray):
        coords1 = [coords1]
    if isinstance(coords2, np.ndarray):
        coords2 = [coords2]
    
    # rotated data
    data_rot = []
    
    # match data coordinates to rotated sphere coordinates, matching is one-to-one!
    if method in ["hungarian", "graph_matching"]:
        for d, c, cr in zip(data, coords1, coords2):
            coord_match = match_coord_sets(
                coords1=c, 
                coords2=cr,
                method=method,
                k=5
            )
            data_rot.append(d[coord_match[:, 1]])

    # distance weighted
    elif method == "distance_weighted":
        for d, c, cr in zip(data, coords1, coords2):
            data_rot.append(
                weighted_average_match(
                    coords1=c, 
                    coords2=cr, 
                    data=d,
                    weight_power=distance_power
                )
            )
    
    # unknown
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # rotated data
    return np.concatenate(data_rot, axis=0), np.concatenate(coords2, axis=0)


def morans_i(data, dist_mat):

    # Inverse distance weights
    with np.errstate(divide='ignore'):
        weight_mat = 1.0 / dist_mat
    np.fill_diagonal(weight_mat, 0.0)  # No self-weighting
    
    # Calculate Moran's I
    n = len(data)
    data_mean = np.mean(data)
    data_diff = data - data_mean
    num = np.sum(weight_mat * np.outer(data_diff, data_diff))
    denom = np.sum(data_diff ** 2)
    morans_i = (n / np.sum(weight_mat)) * (num / denom)
    
    return morans_i

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def optimize_smoothness(data, coords1, coords2, dist_mat=None, initial_smoothness=1.0, 
                        tol=1e-3, method="TNC"):
    """
    Optimize the smoothness parameter to match spatial autocorrelation.

    Parameters:
    - data: A numpy array, or a len(2) list of numpy arrays, of shape (n,) containing the input data values.
    - coords1: A numpy array, or a len(2) list of numpy arrays, of shape (n, 3) containing the input 3D coordinates.
    - coords2: A numpy array, or a len(2) list of numpy arrays, of shape (n, 3) containing the output 3D coordinates.
    - dist_mat: A numpy array of shape (n, n) containing the distance matrix.
    - initial_smoothness: Initial guess for the smoothness parameter.
    - tol: Tolerance for stopping the optimization.

    Returns:
    - Optimized smoothness parameter.
    """
    if isinstance(data, np.ndarray):
        data = [data]
    if isinstance(coords1, np.ndarray):
        coords1 = [coords1]
    if isinstance(coords2, np.ndarray):
        coords2 = [coords2]
    
    if dist_mat is None:
        dist_mat = squareform(pdist(np.concatenate(coords1, axis=0))) 
    
    # target morans i
    target_morans_i = morans_i(np.concatenate(data, axis=0), dist_mat) * 100
    #print(f"Target Moran's I: {target_morans_i:.6f}, tolerance: {tol}")
    
    def objective(smoothness):
        output_data, _ = get_rotated_data(data, coords1, coords2, "distance_weighted", smoothness)
        output_morans_i = morans_i(output_data, dist_mat) * 100
        diff = (target_morans_i - output_morans_i) ** 2
        #print(f"Smoothness: {smoothness}, Moran's I: {output_morans_i:.6f}, Diff: {diff:.6f}") 
        return diff
    
    # Use a simple optimization method like scipy's minimize
    result = minimize(objective, initial_smoothness, bounds=[(0.1, 1000)], method=method, tol=tol)
    print(f"Target Moran's I: {target_morans_i:.6f}, Optimized smoothness: {result.x[0]:.6f}")
    
    if result.success:
        return result.x[0]
    else:
        raise ValueError("Smoothness optimization failed")
    


class VolSpin:
    
    def __init__(self, data, coords=None, parc=None, distmat=None, verbose=True, left_idc=None, right_idc=None):
        
        # check data
        if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series, list)):
            raise ValueError("Data must be a numpy array, pandas DataFrame, pandas Series, or list")
        data = np.array(data).squeeze()
        if data.ndim != 1:
            raise ValueError("Data must be of shape (n,) or (n, 1)")
        
        # check coords and parcellation
        if coords is not None and parc is not None:
            print("Coordinates and parcellation provided, ignoring parcellation")
            parc = None
        elif coords is None and parc is None:
            raise ValueError("Either coordinates or parcellation must be provided")

        # check parc and get coords
        if parc is not None:
            if not isinstance(parc, (nib.Nifti1Image, str, Path)):
                raise ValueError("Parcellation must be (path to) a nibabel Nifti1Image")
            parc = image.load_img(parc)
            if len(data) != len(np.trim_zeros(np.unique(parc.get_fdata()))):
                raise ValueError("Number of indices in parcellation must match number of data points")
            coords = find_vol_parc_centroids(parc)
        
        # check coords
        if coords is not None:
            if not isinstance(coords, (np.ndarray, pd.DataFrame, list)):
                raise ValueError("Coordinates must be a numpy array, pandas DataFrame, or list")
            coords = np.array(coords)
            if coords.shape[0] < 1 or coords.shape[1] != 3:
                raise ValueError("Coordinates must be of shape (n, 3)")
            if len(data) != coords.shape[0]:
                raise ValueError("Data and coordinates must be the same length / dimension 0")
        
        # check distmat
        if distmat is not None:
            if not isinstance(distmat, (np.ndarray, pd.DataFrame)):
                raise ValueError("Distance matrix must be a numpy array or pandas DataFrame")
            distmat = np.array(distmat)
            if distmat.shape[0] != distmat.shape[1] != len(data):
                raise ValueError("Distance matrix must be square and match the length of the data")

        # set
        self._data = data
        self._coords = coords
        self._parc = parc
        self._distmat = distmat
        self._verbose = verbose
        
        
    def fit(self, center=None, radius=None, num_points=100):
        
        # get data center
        if center is None:
            self._center = self._coords.mean(axis=0)

        # get radius
        if radius is None:
            self._radius = largest_distance(self._coords, self._center) * 1.5
        else:
            self._radius = radius
        
        # create the sphere around the data center
        self._sphere_coords = create_sphere_surface(self._radius, num_points, self._center)

        # for each data point, find the closest point on the sphere
        self._data_idc_in_sphere = assign_parcels_to_sphere(self._coords, self._sphere_coords)

        # keep only sphere points with an associated parcel
        self._sphere_coords_data = self._sphere_coords[self._data_idc_in_sphere]
        
        # split data into hemispheres
        if self._parc is not None:
            
            # parcels with center of mass in the left/right hemisphere
            self._data_idc_lh = np.where(self._coords[:, 0] < 0)[0]
            self._data_idc_rh = np.where(self._coords[:, 0] > 0)[0]
        
    
    def transform(self, n_perm=1000, mirrored_rotations=False,
                  method="hungarian", distance_power="auto",
                  seed=None, n_jobs=1):
        
        # optimize smoothness
        opt_smooth = False
        if method == "distance_weighted" and distance_power in ["auto", "optimize"]:
                
            if self._distmat is None:
                self._distmat = squareform(pdist(self._coords))
            opt_smooth = True
            
        # rotation fun
        idc_lh, idc_rh = self._data_idc_lh, self._data_idc_rh 
        def perm_fun(data=self._data, data_coords=self._coords, distmat=self._distmat, distance_power=distance_power,
                     sphere_coords_data=self._sphere_coords_data, center=self._center, seed=seed):
            
            # set random state
            rng = np.random.RandomState(seed)
            
            # rotate the sphere
            sphere_coords_data_rot = rotate_sphere_random(
                sphere_coords=sphere_coords_data, 
                center=center, 
                random_state=rng, 
                axis=None,
                mirror_along_axis="x" if mirrored_rotations else None
            )
            
            # handle mirror
            if isinstance(sphere_coords_data_rot, tuple):
                sphere_coords_data_rot, sphere_coords_data_rot_mirror = sphere_coords_data_rot
            
            # split coords
            if mirrored_rotations:
                data = [
                    data[idc_lh],
                    data[idc_rh]
                ]
                data_coords = [
                    data_coords[idc_lh],
                    data_coords[idc_rh]
                ]
                sphere_coords_data_rot = [
                    sphere_coords_data_rot[idc_lh],
                    sphere_coords_data_rot_mirror[idc_rh]
                ]
            else:
                data = [data]
                data_coords = [data_coords]
                sphere_coords_data_rot = [sphere_coords_data_rot]
                
            # optimize smoothness
            if opt_smooth:
                distance_power = optimize_smoothness(
                    data, 
                    data_coords, 
                    sphere_coords_data_rot, 
                    dist_mat=distmat, 
                    initial_smoothness=100.0, 
                    tol=1e-3
                )

            # rotated data
            data_rot, sphere_coords_data_rot = get_rotated_data(
                data=data, 
                coords1=data_coords, 
                coords2=sphere_coords_data_rot, 
                method=method, 
                distance_power=distance_power
            )
            
            # return
            return data_rot, sphere_coords_data_rot
            
        # run
        data_perm = Parallel(n_jobs=n_jobs)(
            delayed(perm_fun)(seed=(seed+i**2 if seed is not None else None)) 
            for i in tqdm(range(n_perm), disable=not self._verbose)
        )
        self._data_perm = np.stack([d[0] for d in data_perm], axis=0)
        self._sphere_coords_data_perm = np.stack([d[1] for d in data_perm], axis=0)
        
        # return
        return self._data_perm
    
    
    def fit_transform(self, center=None, radius=None, num_points=100, 
                       method="hungarian", n_perm=1000, mirrored_rotations=False,
                       distance_power="auto", seed=None, n_jobs=1):
        self.fit(
            center=center, 
            radius=radius, 
            num_points=num_points
        )
        return self.transform(
            method=method,
            n_perm=n_perm, 
            mirrored_rotations=mirrored_rotations,
            distance_power=distance_power, 
            seed=seed, 
            n_jobs=n_jobs
        )
        

class VolSpin_old:

    def __init__(self, data, coords=None, parc=None, distmat=None, verbose=True):
        
        # check data
        if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series, list)):
            raise ValueError("Data must be a numpy array, pandas DataFrame, pandas Series, or list")
        data = np.array(data).squeeze()
        if data.ndim != 1:
            raise ValueError("Data must be of shape (n,) or (n, 1)")
        
        # check coords and distmat
        if coords is not None and distmat is not None:
            raise ValueError("Coordinates and distance matrix provided, ignoring distance matrix")
            distmat = None
        if coords is not None and parc is not None:
            raise ValueError("Coordinates and parcellation provided, ignoring parcellation")
            parc = None
        if coords is None and distmat is None and parc is None:
            raise ValueError("Either coordinates, distance matrix, or parcellation must be provided")

        # check parc
        if parc is not None:
            if not isinstance(parc, (nib.Nifti1Image, str, Path)):
                raise ValueError("Parcellation must be (path to) a nibabel Nifti1Image")
            parc = image.load_img(parc)
            if len(data) != len(np.trim_zeros(np.unique(parc.get_fdata()))):
                raise ValueError("Number of indices in parcellation must match number of data points")
            coords = find_vol_parc_centroids(parc)
        
        # check distmat
        if distmat is not None:
            if not isinstance(distmat, (np.ndarray, pd.DataFrame)):
                raise ValueError("Distance matrix must be a numpy array or pandas DataFrame")
            distmat = np.array(distmat)
            if distmat.shape[0] != distmat.shape[1] != len(data):
                raise ValueError("Distance matrix must be square and match the length of the data")
            coords = coords_from_distmat(distmat)
             
        # check coords
        if coords is not None:
            if not isinstance(coords, (np.ndarray, pd.DataFrame, list)):
                raise ValueError("Coordinates must be a numpy array, pandas DataFrame, or list")
            coords = np.array(coords)
            if coords.shape[0] < 1 or coords.shape[1] != 3:
                raise ValueError("Coordinates must be of shape (n, 3)")
            if len(data) != coords.shape[0]:
                raise ValueError("Data and coordinates must be the same length / dimension 0")
        
        # set
        self._data = data
        self._coords = coords
        self._parc = parc
        self._distmat = distmat
        self._verbose = verbose
        
    def fit(self, center=None, radius=None, num_points=100):
        
        # get data center
        if center is None:
            self._center = self._coords.mean(axis=0)

        # get radius
        if radius is None:
            self._radius = largest_distance(self._coords, self._center) * 1.5
        else:
            self._radius = radius
        
        # create the sphere around the data center
        self._sphere_coords = create_sphere_surface(self._radius, num_points, self._center)

        # for each data point, find the closest point on the sphere
        self._data_idc_in_sphere = assign_parcels_to_sphere(self._coords, self._sphere_coords)

        # keep only sphere points with an associated parcel
        self._sphere_coords_data = self._sphere_coords[self._data_idc_in_sphere]
        
    def transform(self, n_perm=1000, rotation_axis="x", split_along_axis="x", 
                  mirror_rotation_along_axis="y",
                  method="hungarian", distance_power="auto", 
                  seed=None, n_jobs=1):
        
        # check rotation axis
        if rotation_axis is not None:
            if rotation_axis not in ["x", "y", "z"]:
                raise ValueError("Rotation axis must be 'x', 'y', 'z'")
            
        # optimize smoothness
        if method == "distance_weighted" and distance_power in ["auto", "optimize"]:
            if self._distmat is None:
                self._distmat = squareform(pdist(self._coords))
            
        # prepare split axis
        if split_along_axis is not None:
            if split_along_axis not in ["x", "y", "z"]:
                raise ValueError("Split along axis must be 'x', 'y', 'z'")
            split_along_axis = ["x", "y", "z"].index(split_along_axis)
            
        # prepare mirror axis
        if mirror_rotation_along_axis is not None:
            if mirror_rotation_along_axis not in ["x", "y", "z"]:
                raise ValueError("Mirror rotation along axis must be 'x', 'y', 'z'")            
            
        # rotation fun
        def perm_fun(data=self._data, data_coords=self._coords, distmat=self._distmat, distance_power=distance_power,
                     sphere_coords_data=self._sphere_coords_data, center=self._center, seed=seed):
            
            # set random state
            rng = np.random.RandomState(seed)
            
            # rotate the sphere
            sphere_coords_data_rot = rotate_sphere_random(
                sphere_coords=sphere_coords_data, 
                center=center, 
                random_state=rng, 
                axis=rotation_axis,
                mirror_along_axis=mirror_rotation_along_axis
            )
            
            # handle mirror
            if isinstance(sphere_coords_data_rot, tuple):
                sphere_coords_data_rot, sphere_coords_data_rot_mirror = sphere_coords_data_rot
            
            # optimize smoothness
            if method == "distance_weighted" and distance_power in ["auto", "optimize"]:
                distance_power = optimize_smoothness(
                    data, 
                    data_coords, 
                    sphere_coords_data_rot, 
                    dist_mat=distmat, 
                    initial_smoothness=10.0, 
                    tol=1e-9
                )
            
            # split coords
            if split_along_axis is not None:
                # everything one one/other side of center
                data_bool = data_coords[:, split_along_axis] < center[split_along_axis]
                #data_rot_bool = sphere_coords_data_rot[:, split_along_axis] < center[split_along_axis]
                data = [
                    data[data_bool],
                    data[~data_bool]
                ]
                data_coords = [
                    data_coords[data_bool],
                    data_coords[~data_bool]
                ]
                sphere_coords_data_rot = [
                    sphere_coords_data_rot[data_bool],
                    sphere_coords_data_rot_mirror[~data_bool]
                ]
            else:
                data = [data]
                data_coords = [data_coords]
                sphere_coords_data_rot = [sphere_coords_data_rot]

            # rotated data
            data_rot = []
            # match data coordinates to rotated sphere coordinates, matching is one-to-one!
            if method == "hungarian":
                # first column is index in data_coords, second column is index in sphere_coords_data_rot
                for d, c, cr in zip(data, data_coords, sphere_coords_data_rot):
                    coord_match = match_coord_sets(
                        coords1=c, 
                        coords2=cr
                    )
                    data_rot.append(d[coord_match[:, 1]])
                
            
            # distance weighted
            elif method == "distance_weighted":
                for d, c, cr in zip(data, data_coords, sphere_coords_data_rot):
                    data_rot.append(
                        weighted_average_match(
                            coords1=c, 
                            coords2=cr, 
                            data=d,
                            weight_power=distance_power
                        )
                    )
            
            # rotated data
            return np.concatenate(data_rot, axis=0), np.concatenate(sphere_coords_data_rot, axis=0)
            
        # run
        data_perm = Parallel(n_jobs=n_jobs)(
            delayed(perm_fun)(seed=(seed+i**2 if seed is not None else None)) 
            for i in tqdm(range(n_perm), disable=not self._verbose)
        )
        self._data_perm = np.stack([d[0] for d in data_perm], axis=0)
        self._sphere_coords_data_perm = np.stack([d[1] for d in data_perm], axis=0)
        
        # return
        return self._data_perm
    
    def fit_transform(self, center=None, radius=None, num_points=100, 
                       method="hungarian", n_perm=1000, rotation_axis=None, 
                       distance_power="auto", seed=None, n_jobs=1, split_along_axis="x",
                       mirror_rotation_along_axis=None):
        self.fit(
            center=center, 
            radius=radius, 
            num_points=num_points
        )
        return self.transform(
            method=method,
            n_perm=n_perm, 
            rotation_axis=rotation_axis, 
            distance_power=distance_power, 
            split_along_axis=split_along_axis,
            mirror_rotation_along_axis=mirror_rotation_along_axis,
            seed=seed, 
            n_jobs=n_jobs
        )


def plot_3d_coordinates(coords, title="", colors=None, cmap="viridis", elev=20, azim=30, fig=None, ax=None):
    """
    Plots a set of 3D coordinates in 3D space using Matplotlib.

    Parameters:
    - coords: A list or array of 3D coordinates, where each coordinate is a tuple or list (x, y, z).
    - title: The title of the plot (optional).
    """
    
    if ax is None:
        # Create a new figure for the 3D plot
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    # Unpack the coordinates
    x, y, z = zip(*coords)
    # Plot the points
    ax.scatter(x, y, z, c=colors, marker='o', alpha=0.7, cmap=cmap)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set the camera position
    ax.view_init(elev=elev, azim=azim)

    # return
    return fig, ax
