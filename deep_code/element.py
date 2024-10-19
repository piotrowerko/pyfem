from dataclasses import dataclass, field
import numpy as np
from .node import Node
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Element:
    id: int
    node_start: Node
    node_end: Node
    E: float       # Young's Modulus
    G: float       # Shear Modulus
    A: float       # Cross-sectional area
    Iy: float      # Second moment of area about y-axis
    Iz: float      # Second moment of area about z-axis
    J: float       # Torsional constant
    ky: float      # Shear correction factor in y-direction
    kz: float      # Shear correction factor in z-direction
    length: float = field(init=False)
    local_stiffness_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Calculate the length of the element
        self.length = np.linalg.norm(self.node_end.coordinates - self.node_start.coordinates)
        # Initialize the local stiffness matrix
        self.local_stiffness_matrix = self.compute_local_stiffness_matrix_euler()

    def compute_local_stiffness_matrix_timoshenko(self)  -> np.ndarray:
        """Compute the local stiffness matrix using Timoshenko beam theory."""
        L = self.length
        E = self.E
        G = self.G
        A = self.A
        Iy = self.Iy
        Iz = self.Iz
        J = self.J
        ky = self.ky
        kz = self.kz

        # Shear coefficients
        phi_y = (12 * E * Iz) / (kz * A * G * L ** 2)
        phi_z = (12 * E * Iy) / (ky * A * G * L ** 2)

        # Stiffness matrix components
        # Axial stiffness
        k_axial = E * A / L

        # Torsional stiffness
        k_torsion = G * J / L

        # Bending stiffness about y-axis
        k_bend_y = E * Iy / (L * (1 + phi_z))
        # Bending stiffness about z-axis
        k_bend_z = E * Iz / (L * (1 + phi_y))

        # Shear stiffness
        k_shear_y = k_bend_y * phi_z
        k_shear_z = k_bend_z * phi_y

        # Initialize 12x12 stiffness matrix as NumPy array
        k_local = np.zeros((12, 12))

        # Degrees of freedom mapping
        # [uX1, uY1, uZ1, θX1, θY1, θZ1, uX2, uY2, uZ2, θX2, θY2, θZ2]

        # Assemble the stiffness matrix using NumPy operations

        # Axial terms
        axial_indices = np.array([0, 6])
        k_axial_matrix = np.array([[ k_axial, -k_axial],
                                   [-k_axial,  k_axial]])
        k_local[np.ix_(axial_indices, axial_indices)] += k_axial_matrix

        # Torsional terms
        torsion_indices = np.array([3, 9])
        k_torsion_matrix = np.array([[ k_torsion, -k_torsion],
                                     [-k_torsion,  k_torsion]])
        k_local[np.ix_(torsion_indices, torsion_indices)] += k_torsion_matrix

        # Bending about y-axis (affecting z-direction displacements and rotations about y-axis)
        indices_z = np.array([2, 4, 8, 10])
        k_bending_z = np.array([
            [ k_shear_y + (12 * k_bend_y) / L**2,   k_shear_y * (L / 2) + (6 * k_bend_y) / L, -k_shear_y - (12 * k_bend_y) / L**2,   k_shear_y * (L / 2) - (6 * k_bend_y) / L],
            [ k_shear_y * (L / 2) + (6 * k_bend_y) / L, k_shear_y * (L**2 / 4) + (4 * k_bend_y), -k_shear_y * (L / 2) + (6 * k_bend_y) / L, k_shear_y * (L**2 / 4) - (2 * k_bend_y)],
            [-k_shear_y - (12 * k_bend_y) / L**2,  -k_shear_y * (L / 2) + (6 * k_bend_y) / L,  k_shear_y + (12 * k_bend_y) / L**2,  -k_shear_y * (L / 2) - (6 * k_bend_y) / L],
            [ k_shear_y * (L / 2) - (6 * k_bend_y) / L, k_shear_y * (L**2 / 4) - (2 * k_bend_y), -k_shear_y * (L / 2) - (6 * k_bend_y) / L, k_shear_y * (L**2 / 4) + (4 * k_bend_y)]
        ])
        k_local[np.ix_(indices_z, indices_z)] += k_bending_z

        # Bending about z-axis (affecting y-direction displacements and rotations about z-axis)
        indices_y = np.array([1, 5, 7, 11])
        k_bending_y = np.array([
            [ k_shear_z + (12 * k_bend_z) / L**2,  -k_shear_z * (L / 2) + (6 * k_bend_z) / L, -k_shear_z - (12 * k_bend_z) / L**2,  -k_shear_z * (L / 2) - (6 * k_bend_z) / L],
            [-k_shear_z * (L / 2) + (6 * k_bend_z) / L, k_shear_z * (L**2 / 4) + (4 * k_bend_z),  k_shear_z * (L / 2) - (6 * k_bend_z) / L, k_shear_z * (L**2 / 4) - (2 * k_bend_z)],
            [-k_shear_z - (12 * k_bend_z) / L**2,  k_shear_z * (L / 2) - (6 * k_bend_z) / L,  k_shear_z + (12 * k_bend_z) / L**2,   k_shear_z * (L / 2) + (6 * k_bend_z) / L],
            [-k_shear_z * (L / 2) - (6 * k_bend_z) / L, k_shear_z * (L**2 / 4) - (2 * k_bend_z),  k_shear_z * (L / 2) + (6 * k_bend_z) / L, k_shear_z * (L**2 / 4) + (4 * k_bend_z)]
        ])
        k_local[np.ix_(indices_y, indices_y)] += k_bending_y

        return k_local
    
    def compute_local_stiffness_matrix_euler(self) -> np.ndarray:
        """Compute the local stiffness matrix using Euler-Bernoulli beam theory with NumPy operations."""
        L = self.length
        E = self.E
        G = self.G
        A = self.A
        Iy = self.Iy
        Iz = self.Iz
        J = self.J

        # Axial stiffness
        k_axial = E * A / L

        # Torsional stiffness
        k_torsion = G * J / L

        # Bending stiffness about y-axis
        k_bend_y = E * Iy / L**3

        # Bending stiffness about z-axis
        k_bend_z = E * Iz / L**3

        # Initialize 12x12 stiffness matrix as NumPy array
        k_local = np.zeros((12, 12))

        # Degrees of freedom mapping
        # [uX1, uY1, uZ1, θX1, θY1, θZ1, uX2, uY2, uZ2, θX2, θY2, θZ2]

        # Axial terms
        axial_dofs = [0, 6]
        k_axial_matrix = k_axial * np.array([
            [1, -1],
            [-1, 1]
        ])
        idx = np.ix_(axial_dofs, axial_dofs)
        k_local[idx] += k_axial_matrix

        # Torsional terms
        torsion_dofs = [3, 9]
        k_torsion_matrix = k_torsion * np.array([
            [1, -1],
            [-1, 1]
        ])
        idx = np.ix_(torsion_dofs, torsion_dofs)
        k_local[idx] += k_torsion_matrix

        # Bending about y-axis (affecting z-displacements and rotations about y-axis)
        bending_y_dofs = [2, 4, 8, 10]
        k_bending_y_matrix = k_bend_y * np.array([
            [12,    6*L,   -12,    6*L],
            [6*L,  4*L**2, -6*L,  2*L**2],
            [-12,  -6*L,    12,   -6*L],
            [6*L,  2*L**2, -6*L,  4*L**2]
        ])
        idx = np.ix_(bending_y_dofs, bending_y_dofs)
        k_local[idx] += k_bending_y_matrix

        # Bending about z-axis (affecting y-displacements and rotations about z-axis)
        bending_z_dofs = [1, 5, 7, 11]
        k_bending_z_matrix = k_bend_z * np.array([
            [12,   -6*L,   -12,   -6*L],
            [-6*L, 4*L**2, 6*L,   2*L**2],
            [-12,  6*L,    12,    6*L],
            [-6*L, 2*L**2, 6*L,   4*L**2]
        ])
        idx = np.ix_(bending_z_dofs, bending_z_dofs)
        k_local[idx] += k_bending_z_matrix

        return k_local


