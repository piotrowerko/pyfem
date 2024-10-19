from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from .element import Element
from .node import Node
from .transformer import transform_stiffness_matrix

@dataclass
class StructuralElements:
    elements: List[Element] = field(default_factory=list)
    nodes: Dict[int, Node] = field(default_factory=dict)
    global_stiffness_matrix: np.ndarray = field(init=False, repr=False)
    nodal_displacements: np.ndarray = field(init=False, repr=False)
    nodal_forces: np.ndarray = field(init=False, repr=False)
    total_dofs: int = field(init=False)

    def __post_init__(self):
        # Initialize total degrees of freedom
        self.total_dofs = len(self.nodes) * 6  # 6 DOFs per node
        # Initialize global stiffness matrix and vectors
        self.global_stiffness_matrix = np.zeros((self.total_dofs, self.total_dofs))
        self.nodal_displacements = np.zeros(self.total_dofs)
        self.nodal_forces = np.zeros(self.total_dofs)

    def add_node(self, node: Node):
        """Adds a node to the structure."""
        self.nodes[node.id] = node
        # Update total degrees of freedom
        self.total_dofs = len(self.nodes) * 6
        # Resize global matrices and vectors accordingly
        self.global_stiffness_matrix = np.zeros((self.total_dofs, self.total_dofs))
        self.nodal_displacements = np.zeros(self.total_dofs)
        self.nodal_forces = np.zeros(self.total_dofs)

    def add_element(self, element: Element):
        """Adds an element to the structure."""
        self.elements.append(element)

    def assemble_global_stiffness_matrix(self):
        """
        Assembles the global stiffness matrix from all elements using NumPy operations.
        """
        # Reset the global stiffness matrix
        self.global_stiffness_matrix.fill(0)
        # Iterate over all elements
        for element in self.elements:
            # Transform the local stiffness matrix to global coordinates
            k_global = transform_stiffness_matrix(element)
            # Get the DOF indices for the element
            dof_indices = self.get_element_dof_indices(element)
            # Assemble into the global stiffness matrix using NumPy advanced indexing
            idx = np.ix_(dof_indices, dof_indices)
            self.global_stiffness_matrix[idx] += k_global
        # Store a copy of the original stiffness matrix
        self.original_global_stiffness_matrix = self.global_stiffness_matrix.copy()


    def get_element_dof_indices(self, element: Element) -> List[int]:
        """Returns the global DOF indices for the given element."""
        node_ids = [element.node_start.id, element.node_end.id]
        dof_indices = []
        for node_id in node_ids:
            base_index = (node_id - 1) * 6  # Adjusting for zero-based indexing
            dof_indices.extend([base_index + i for i in range(6)])
        return dof_indices

    def apply_boundary_conditions(self, fixed_dofs: List[int]):
        """
        Applies boundary conditions by modifying the global stiffness matrix.
        """
        for dof in fixed_dofs:
            # Zero out the rows and columns for the fixed DOFs
            self.global_stiffness_matrix[dof, :] = 0
            self.global_stiffness_matrix[:, dof] = 0
            # Set diagonal terms to 1 to prevent singular matrix
            self.global_stiffness_matrix[dof, dof] = 1
            # Do not modify the nodal forces at fixed DOFs
            # Do not modify self.nodal_forces[dof]


    def solve(self):
        """
        Solves the system of equations to find nodal displacements.
        """
        self.nodal_displacements = np.linalg.solve(self.global_stiffness_matrix, self.nodal_forces)

    def get_reactions(self) -> np.ndarray:
        """
        Calculates reactions at supports using the original stiffness matrix.
        """
        reactions = self.original_global_stiffness_matrix @ self.nodal_displacements - self.nodal_forces
        return reactions


