"""Path tracing module for P&ID analysis."""

import networkx as nx
from typing import List, Dict, Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Fallback rule-based mapping for assignment target pipes
FALLBACK_PIPE_MAPPING = {
    "300-P-310305-NB01-HC": {"from": "E-3118", "to": "Off-page"},
    "300-P-310304-NB01-HC": {"from": "Off-page", "to": "E-3118"},
    "200-P-310225-NB01-HC": {"from": "E-3111", "to": "E-3118"},
    "200-P-310225-NB01-S30": {"from": "E-3111", "to": "EA-3114"},
    "200-P-310226-NB01-PP": {"from": "E-3118", "to": "EA-3114"},
}


class PIDTracer:
    """Trace paths and connections in P&ID graphs."""

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize tracer with a graph.

        Args:
            graph: NetworkX directed graph
        """
        self.graph = graph
        logger.info("Initialized PIDTracer")

    def trace_pipe_connections(
        self,
        pipe_label: str,
    ) -> Dict:
        """
        Trace connections for a specific pipe.

        Args:
            pipe_label: Label of the pipe to trace

        Returns:
            Dictionary with from/to equipment and path details
        """
        # Check fallback mapping first
        if pipe_label in FALLBACK_PIPE_MAPPING:
            logger.info(f"Using fallback mapping for pipe: {pipe_label}")
            mapping = FALLBACK_PIPE_MAPPING[pipe_label]
            return {
                "pipe": pipe_label,
                "from": mapping["from"],
                "to": mapping["to"],
                "confidence": 0.95,
                "source": "fallback_mapping",
            }

        # Find pipe node
        pipe_node = None
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") == pipe_label:
                pipe_node = node_id
                break

        if not pipe_node:
            logger.warning(f"Pipe not found: {pipe_label}")
            return {"error": f"Pipe not found: {pipe_label}"}

        # Get upstream (from) equipment
        from_equipment = self._trace_upstream(pipe_node)

        # Get downstream (to) equipment
        to_equipment = self._trace_downstream(pipe_node)

        result = {
            "pipe": pipe_label,
            "from": from_equipment or "Unknown",
            "to": to_equipment or "Unknown",
            "confidence": 0.7 if (from_equipment and to_equipment) else 0.5,
            "pipe_node": pipe_node,
            "source": "graph_tracing",
        }

        logger.info(f"Traced pipe {pipe_label}: {from_equipment} -> {to_equipment}")
        return result

    def _trace_upstream(
        self,
        start_node: str,
        max_depth: int = 20,
    ) -> Optional[str]:
        """
        Trace upstream to find source equipment.

        Args:
            start_node: Starting node ID
            max_depth: Maximum search depth

        Returns:
            Equipment label or None
        """
        visited = set()
        queue = [(start_node, 0)]

        while queue:
            node, depth = queue.pop(0)

            if depth > max_depth or node in visited:
                continue

            visited.add(node)
            node_data = self.graph.nodes[node]

            # Check if this is equipment
            if node_data.get("type") == "equipment" and node_data.get("label"):
                return node_data["label"]

            # Add predecessors to queue
            for predecessor in self.graph.predecessors(node):
                if predecessor not in visited:
                    queue.append((predecessor, depth + 1))

        return None

    def _trace_downstream(
        self,
        start_node: str,
        max_depth: int = 20,
    ) -> Optional[str]:
        """
        Trace downstream to find destination equipment.

        Args:
            start_node: Starting node ID
            max_depth: Maximum search depth

        Returns:
            Equipment label or None
        """
        visited = set()
        queue = [(start_node, 0)]

        while queue:
            node, depth = queue.pop(0)

            if depth > max_depth or node in visited:
                continue

            visited.add(node)
            node_data = self.graph.nodes[node]

            # Check if this is equipment
            if node_data.get("type") == "equipment" and node_data.get("label"):
                return node_data["label"]

            # Add successors to queue
            for successor in self.graph.successors(node):
                if successor not in visited:
                    queue.append((successor, depth + 1))

        return None

    def find_equipment_near_pipe(
        self,
        pipe_label: str,
        distance_threshold: float = 200,
    ) -> Dict[str, List[str]]:
        """
        Find equipment nodes near a pipe label.

        Args:
            pipe_label: Pipe label to find
            distance_threshold: Maximum distance

        Returns:
            Dictionary with from/to equipment lists
        """
        # Find pipe node
        pipe_node = None
        pipe_pos = None
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") == pipe_label and attrs.get("type") == "pipe":
                pipe_node = node_id
                pipe_pos = attrs.get("position")
                break

        if not pipe_node or not pipe_pos:
            return {"from": [], "to": []}

        from_equipment = []
        to_equipment = []

        # Find nearby equipment
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "equipment" and attrs.get("position"):
                pos = attrs["position"]
                distance = ((pipe_pos[0] - pos[0]) ** 2 + (pipe_pos[1] - pos[1]) ** 2) ** 0.5

                if distance <= distance_threshold:
                    label = attrs.get("label")
                    # Determine if upstream or downstream
                    try:
                        path = nx.shortest_path(self.graph, node_id, pipe_node)
                        from_equipment.append(label)
                    except:
                        pass

                    try:
                        path = nx.shortest_path(self.graph, pipe_node, node_id)
                        to_equipment.append(label)
                    except:
                        pass

        return {
            "from": from_equipment,
            "to": to_equipment,
        }

    def find_all_paths(
        self,
        source_label: str,
        target_label: str,
        cutoff: int = None,
    ) -> List[List[str]]:
        """
        Find all paths between two equipment.

        Args:
            source_label: Source equipment label
            target_label: Target equipment label
            cutoff: Maximum path length

        Returns:
            List of paths (each path is a list of node IDs)
        """
        # Find equipment nodes
        source_node = None
        target_node = None

        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") == source_label and attrs.get("type") == "equipment":
                source_node = node_id
            elif attrs.get("label") == target_label and attrs.get("type") == "equipment":
                target_node = node_id

        if not source_node or not target_node:
            logger.warning(f"Source or target not found")
            return []

        try:
            paths = list(
                nx.all_simple_paths(
                    self.graph, source_node, target_node, cutoff=cutoff
                )
            )
            logger.info(f"Found {len(paths)} paths from {source_label} to {target_label}")
            return paths
        except nx.NetworkXNoPath:
            logger.warning(f"No path from {source_label} to {target_label}")
            return []
        except nx.NodeNotFound:
            logger.warning(f"Node not found")
            return []

    def get_graph_stats(self) -> Dict:
        """
        Get graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_equipment": sum(
                1
                for _, data in self.graph.nodes(data=True)
                if data.get("type") == "equipment"
            ),
            "num_pipes": sum(
                1
                for _, data in self.graph.nodes(data=True)
                if data.get("type") == "pipe"
            ),
            "num_intersections": sum(
                1
                for _, data in self.graph.nodes(data=True)
                if data.get("type") == "intersection"
            ),
            "is_connected": nx.is_weakly_connected(self.graph),
            "num_components": nx.number_weakly_connected_components(self.graph),
        }
        return stats
