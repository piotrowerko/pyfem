from dataclasses import dataclass, field
import numpy as np

@dataclass
class Node:
    id: int
    coordinates: np.ndarray = field(default_factory=lambda: np.zeros(3))
    displacements: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotations: np.ndarray = field(default_factory=lambda: np.zeros(3))