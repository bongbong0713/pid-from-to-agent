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
        self.pipe_nodes = {}  # Map pipe label -> node
        self.line_nodes = {}  # Map line segment -> node
        self.intersection_nodes = {}  # Map (x, y) -> node
        self.node_counter = 0
        logger.info("Initialized GraphBuilder")

    def _get_unique_node_id(self, prefix: str = "node") -> str:
        """
        Generate unique node ID.

        Args:
            prefix: Node ID prefix

        Returns:
            Unique node ID
        """
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def add_equipment_node(
        self,
        label: str,
        node_type: str = "equipment",
        position: Tuple[float, float] = None,
        **metadata,
    ) -> str:
        """
        Add an equipment node to the graph.

        Args:
            label: Equipment label
            node_type: Type of node (equipment, valve, etc.)
            position: (x, y) coordinates
            **metadata: Additional metadata

        Returns:
            Node ID
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
        return node_id

    def add_pipe_node(
        self,
        label: str,
        position: Tuple[float, float],
        bbox: Dict[str, float] = None,
        node_type: str = "pipe",
        **metadata,
    ) -> str:
        """
        Add a pipe label node to the graph.

        Args:
            label: Pipe label
            position: (x, y) center coordinates
            bbox: Bounding box information
            node_type: Type of node
            **metadata: Additional metadata

        Returns:
            Node ID
        """
        node_id = f"pipe_{label}"
        self.graph.add_node(
            node_id,
            label=label,
            type=node_type,
            position=position,
            bbox=bbox,
            **metadata,
        )
        self.pipe_nodes[label] = node_id
        logger.info(f"Added pipe node: {label}")
        return node_id

    def add_line_node(
        self,
        position: Tuple[float, float],
        node_type: str = "line",
        line_segment: Tuple[Tuple[int, int], Tuple[int, int]] = None,
        **metadata,
    ) -> str:
        """
        Add a line segment node to the graph.

        Args:
            position: (x, y) center coordinates
            node_type: Type of node
            line_segment: The line segment coordinates
            **metadata: Additional metadata

        Returns:
            Node ID
        """
        node_id = self._get_unique_node_id("line")
        self.graph.add_node(
            node_id,
            label=None,
            type=node_type,
            position=position,
            line_segment=line_segment,
            **metadata,
        )
        return node_id

    def add_intersection_node(
        self,
        position: Tuple[float, float],
        node_type: str = "intersection",
        **metadata,
    ) -> str:
        """
        Add an intersection node to the graph.

        Args:
            position: (x, y) coordinates
            node_type: Type of node
            **metadata: Additional metadata

        Returns:
            Node ID
        """
        node_id = f"int_{position[0]:.1f}_{position[1]:.1f}"
        if node_id not in self.graph:
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

    def connect_pipe_to_lines(
        self,
        pipe_label: str,
        lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        distance_threshold: float = 100,
    ):
        """
        Connect pipe label node to nearest line segments.

        Args:
            pipe_label: Pipe label
            lines: List of line segments
            distance_threshold: Maximum connection distance
        """
        if pipe_label not in self.pipe_nodes:
            logger.warning(f"Pipe node not found: {pipe_label}")
            return

        pipe_node_id = self.pipe_nodes[pipe_label]
        pipe_node = self.graph.nodes[pipe_node_id]
        pipe_pos = pipe_node["position"]

        connected_count = 0
        for line in lines:
            (x1, y1), (x2, y2) = line
            line_center_x = (x1 + x2) / 2
            line_center_y = (y1 + y2) / 2

            # Calculate distance from pipe to line center
            distance = np.sqrt(
                (pipe_pos[0] - line_center_x) ** 2 + (pipe_pos[1] - line_center_y) ** 2
            )

            if distance <= distance_threshold:
                # Create line node
                line_node_id = self.add_line_node(
                    (line_center_x, line_center_y),
                    line_segment=line,
                )
                # Connect pipe to line
                self.add_edge(pipe_node_id, line_node_id, edge_type="label_to_pipe")
                self.add_edge(line_node_id, pipe_node_id, edge_type="pipe_to_label")
                connected_count += 1

        logger.info(f"Connected pipe {pipe_label} to {connected_count} line segments")

    def connect_line_endpoints(
        self,
        lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    ):
        """
        Connect line segment endpoints and intersections.

        Args:
            lines: List of line segments
        """
        # Create nodes for line endpoints
        endpoint_nodes = {}
        for line in lines:
            (x1, y1), (x2, y2) = line
            # Add endpoint nodes
            key1 = (x1, y1)
            key2 = (x2, y2)

            if key1 not in endpoint_nodes:
                node_id = self.add_intersection_node(key1)
                endpoint_nodes[key1] = node_id
            if key2 not in endpoint_nodes:
                node_id = self.add_intersection_node(key2)
                endpoint_nodes[key2] = node_id

            # Connect endpoints
            self.add_edge(
                endpoint_nodes[key1],
                endpoint_nodes[key2],
                edge_type="line_segment",
                distance=np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
            )
            self.add_edge(
                endpoint_nodes[key2],
                endpoint_nodes[key1],
                edge_type="line_segment",
                distance=np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
            )

        logger.info(f"Connected line endpoints for {len(lines)} line segments")

    def build_from_detection(
        self,
        equipment: List[Dict[str, Any]],
        lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        intersections: List[Tuple[float, float]],
        pipe_labels: List[Dict[str, Any]] = None,
    ):
        """
        Build graph from detection results.

        Args:
            equipment: List of detected equipment
            lines: List of detected line segments
            intersections: List of detected intersections
            pipe_labels: List of detected pipe labels (optional)
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

        # Add pipe label nodes
        if pipe_labels:
            for pipe in pipe_labels:
                position = (
                    pipe["bbox"]["center_x"],
                    pipe["bbox"]["center_y"],
                )
                self.add_pipe_node(
                    pipe["text"],
                    position=position,
                    bbox=pipe["bbox"],
                )

        # Add intersection nodes
        for intersection in intersections:
            self.add_intersection_node(intersection)

        # Connect line endpoints
        self.connect_line_endpoints(lines)

        # Connect pipe labels to nearest line segments
        if pipe_labels:
            for pipe in pipe_labels:
                self.connect_pipe_to_lines(pipe["text"], lines)

        # Connect equipment to nearby line endpoints/intersections
        for label, eq_node_id in self.equipment_nodes.items():
            eq_node = self.graph.nodes[eq_node_id]
            eq_pos = eq_node["position"]

            # Find nearest line intersections/endpoints
            for int_pos, int_node_id in self.intersection_nodes.items():
                distance = np.sqrt(
                    (eq_pos[0] - int_pos[0]) ** 2 + (eq_pos[1] - int_pos[1]) ** 2
                )
                if distance < 200:  # Threshold: 200 pixels
                    self.add_edge(eq_node_id, int_node_id, edge_type="connection")
                    self.add_edge(int_node_id, eq_node_id, edge_type="connection")

        logger.info(
            f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )

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
                    **{k: v for k, v in self.graph.nodes[node].items() if k != "bbox"},
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
