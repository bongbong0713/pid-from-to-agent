"""NetworkX graph construction for P&ID analysis."""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GraphBuilder:
    """Build and manage P&ID network graphs."""

    def __init__(self):
        """
        Initialize graph builder.
        """
        self.graph = nx.DiGraph()  # Directed graph
        self.equipment_nodes = {}  # Map label -> node
        self.intersection_nodes = {}  # Map (x, y) -> node
        logger.info("Initialized GraphBuilder")

    def add_equipment_node(
        self,
        label: str,
        node_type: str = "equipment",
        position: Tuple[float, float] = None,
        **metadata,
    ):
        """
        Add an equipment node to the graph.

        Args:
            label: Equipment label
            node_type: Type of node (equipment, valve, etc.)
            position: (x, y) coordinates
            **metadata: Additional metadata
        """
        node_id = f"eq_{label}"
        self.graph.add_node(
            node_id,
            label=label,
            type=node_type,
            position=position,
            **metadata,
        )
        self.equipment_nodes[label] = node_id
        logger.info(f"Added equipment node: {label}")

    def add_intersection_node(
        self,
        position: Tuple[float, float],
        node_type: str = "intersection",
        **metadata,
    ):
        """
        Add an intersection node to the graph.

        Args:
            position: (x, y) coordinates
            node_type: Type of node
            **metadata: Additional metadata
        """
        node_id = f"int_{position[0]:.1f}_{position[1]:.1f}"
        self.graph.add_node(
            node_id,
            label=None,
            type=node_type,
            position=position,
            **metadata,
        )
        self.intersection_nodes[position] = node_id
        return node_id

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: str = "pipe",
        **metadata,
    ):
        """
        Add an edge to the graph.

        Args:
            from_node: Source node ID
            to_node: Destination node ID
            edge_type: Type of edge (pipe, valve, etc.)
            **metadata: Additional metadata
        """
        self.graph.add_edge(
            from_node,
            to_node,
            type=edge_type,
            **metadata,
        )
        logger.info(f"Added edge: {from_node} -> {to_node}")

    def build_from_detection(
        self,
        equipment: List[Dict[str, Any]],
        lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        intersections: List[Tuple[float, float]],
    ):
        """
        Build graph from detection results.

        Args:
            equipment: List of detected equipment
            lines: List of detected line segments
            intersections: List of detected intersections
        """
        # Add equipment nodes
        for eq in equipment:
            position = (
                eq["bbox"]["center_x"],
                eq["bbox"]["center_y"],
            )
            self.add_equipment_node(
                eq["text"],
                node_type=eq.get("type", "equipment"),
                position=position,
            )

        # Add intersection nodes
        for intersection in intersections:
            self.add_intersection_node(intersection)

        # Connect equipment to nearby intersections
        for label, eq_node_id in self.equipment_nodes.items():
            eq_node = self.graph.nodes[eq_node_id]
            eq_pos = eq_node["position"]

            # Find nearest intersections
            for int_pos, int_node_id in self.intersection_nodes.items():
                distance = np.sqrt(
                    (eq_pos[0] - int_pos[0]) ** 2 + (eq_pos[1] - int_pos[1]) ** 2
                )
                if distance < 100:  # Threshold: 100 pixels
                    self.add_edge(eq_node_id, int_node_id, edge_type="connection")
                    self.add_edge(int_node_id, eq_node_id, edge_type="connection")

        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

    def find_path(
        self,
        source_label: str,
        target_label: str,
    ) -> List[str]:
        """
        Find path between two equipment nodes.

        Args:
            source_label: Source equipment label
            target_label: Target equipment label

        Returns:
            List of node IDs in the path, or empty list if no path exists
        """
        if source_label not in self.equipment_nodes:
            logger.warning(f"Source equipment not found: {source_label}")
            return []

        if target_label not in self.equipment_nodes:
            logger.warning(f"Target equipment not found: {target_label}")
            return []

        source_node = self.equipment_nodes[source_label]
        target_node = self.equipment_nodes[target_label]

        try:
            path = nx.shortest_path(self.graph, source_node, target_node)
            logger.info(f"Found path from {source_label} to {target_label}: {len(path)} nodes")
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"No path found from {source_label} to {target_label}")
            return []
        except nx.NodeNotFound:
            logger.error(f"Node not found in graph")
            return []

    def get_connected_components(self) -> List[List[str]]:
        """
        Get all connected components in the graph.

        Returns:
            List of connected component node lists
        """
        components = list(nx.weakly_connected_components(self.graph))
        logger.info(f"Found {len(components)} connected components")
        return [list(comp) for comp in components]

    def get_neighbors(
        self,
        label: str,
        direction: str = "both",
    ) -> List[str]:
        """
        Get neighbor nodes.

        Args:
            label: Equipment label
            direction: 'in', 'out', or 'both'

        Returns:
            List of neighbor node IDs
        """
        if label not in self.equipment_nodes:
            return []

        node_id = self.equipment_nodes[label]

        if direction == "in":
            neighbors = list(self.graph.predecessors(node_id))
        elif direction == "out":
            neighbors = list(self.graph.successors(node_id))
        else:  # both
            neighbors = list(self.graph.neighbors(node_id)) + list(
                self.graph.predecessors(node_id)
            )

        return neighbors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to dictionary representation.

        Returns:
            Dictionary with nodes and edges
        """
        return {
            "nodes": [
                {
                    "id": node,
                    **self.graph.nodes[node],
                }
                for node in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **self.graph[u][v],
                }
                for u, v in self.graph.edges()
            ],
        }
