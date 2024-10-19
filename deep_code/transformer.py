from .element import Element
import numpy as np

def transform_stiffness_matrix(element: Element) -> np.ndarray:
    """
    Transforms the local stiffness matrix to the global coordinate system based on the orientation of the element.

    Parameters:
    - element: An instance of the Element class containing the local stiffness matrix and node coordinates.

    Returns:
    - k_global: The global stiffness matrix as a NumPy array.
    """
    # Extract node coordinates
    x1, y1, z1 = element.node_start.coordinates
    x2, y2, z2 = element.node_end.coordinates

    # Element length
    L = element.length

    # Direction cosines of the element (local x-axis)
    lx = (x2 - x1) / L
    ly = (y2 - y1) / L
    lz = (z2 - z1) / L

    x_local = np.array([lx, ly, lz])

    # Define a reference vector not parallel to x_local
    # Choose global Y-axis if x_local is not parallel to it
    if not np.allclose(x_local, [0, 1, 0]):
        v = np.array([0, 1, 0])
    else:
        # Use global Z-axis as reference vector
        v = np.array([0, 0, 1])

    # Compute local z-axis (z_local) as cross product of x_local and v
    z_local = np.cross(x_local, v)
    z_local /= np.linalg.norm(z_local)

    # Compute local y-axis (y_local) as cross product of z_local and x_local
    y_local = np.cross(z_local, x_local)
    y_local /= np.linalg.norm(y_local)

    # Assemble the rotation matrix R (3x3)
    R = np.vstack([x_local, y_local, z_local])

    # Build the transformation matrix T (12x12)
    T = np.zeros((12, 12))

    # Populate the T matrix with R in appropriate positions
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    T[6:9, 6:9] = R
    T[9:12, 9:12] = R

    # Transform the local stiffness matrix to global coordinates
    k_local = element.local_stiffness_matrix

    # Global stiffness matrix
    k_global = T.T @ k_local @ T

    return k_global
