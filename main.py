# main.py

import numpy as np
from deep_code.node import Node
from deep_code.element import Element
from deep_code.structure import StructuralElements

def main():
    # Define nodes
    node1 = Node(id=1, coordinates=np.array([0.0, 0.0, 0.0]))     # Left support
    node2 = Node(id=2, coordinates=np.array([0.5, 0.0, 0.0]))     # Midpoint
    node3 = Node(id=3, coordinates=np.array([1.0, 0.0, 0.0]))     # Right support

    # Create StructuralElements instance
    structure = StructuralElements()
    structure.add_node(node1)
    structure.add_node(node2)
    structure.add_node(node3)

    # Material and section properties
    E = 210e9       # Young's Modulus in Pascals
    G = 81.2e9      # Shear Modulus in Pascals
    A = 0.005       # Cross-sectional area in m^2
    Iy = 8.333e-6   # Second moment of area about y-axis in m^4
    Iz = 8.333e-6   # Second moment of area about z-axis in m^4
    J = 1.666e-5    # Torsional constant in m^4
    ky = 5/6        # Shear correction factor in y-direction
    kz = 5/6        # Shear correction factor in z-direction

    # Define elements
    element1 = Element(
        id=1,
        node_start=node1,
        node_end=node2,
        E=E,
        G=G,
        A=A,
        Iy=Iy,
        Iz=Iz,
        J=J,
        ky=ky,
        kz=kz
    )

    element2 = Element(
        id=2,
        node_start=node2,
        node_end=node3,
        E=E,
        G=G,
        A=A,
        Iy=Iy,
        Iz=Iz,
        J=J,
        ky=ky,
        kz=kz
    )

    # Add elements to structure
    structure.add_element(element1)
    structure.add_element(element2)

    # Assemble global stiffness matrix
    structure.assemble_global_stiffness_matrix()

    # Apply external nodal force at the midpoint (Node 2)
    P = -10000.0  # Load applied at the midpoint (negative for downward force)

    # Apply force to nodal forces vector at Node 2
    dof_index_node2 = (node2.id - 1) * 6 + 1  # uY DOF of Node 2
    structure.nodal_forces[dof_index_node2] = P  # Apply load in Y-direction

    # Apply boundary conditions
    # Fixing all DOFs at Node 1 and Node 3
    fixed_dofs = []
    for node_id in [node1.id, node3.id]:
        base_index = (node_id - 1) * 6
        fixed_dofs.extend([base_index + i for i in range(6)])
    structure.apply_boundary_conditions(fixed_dofs)

    # Solve for nodal displacements
    structure.solve()

    # Print nodal displacements
    print("Nodal Displacements:")
    for node_id, node in structure.nodes.items():
        base_index = (node_id - 1) * 6
        displacements = structure.nodal_displacements[base_index:base_index + 6]
        print(f"Node {node_id}: {displacements}")

    # Calculate and print support reactions
    reactions = structure.get_reactions()
    print("\nSupport Reactions at Fixed DOFs:")
    for dof in fixed_dofs:
        node_id = dof // 6 + 1
        dof_type = ['uX', 'uY', 'uZ', 'θX', 'θY', 'θZ'][dof % 6]
        reaction = reactions[dof]
        print(f"Node {node_id}, DOF {dof_type}: Reaction = {reaction:.2f} N")

if __name__ == "__main__":
    main()
