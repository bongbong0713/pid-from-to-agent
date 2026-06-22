"""Main LangChain agent for P&ID analysis."""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import cv2
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.state import START

from src.utils.logger import setup_logger
from src.ocr.paddle_ocr import PaddleOCRExtractor
from src.vision.preprocessing import ImagePreprocessor
from src.vision.pipe_detector import PipeDetector
from src.vision.equipment_detector import EquipmentDetector
from src.graph.graph_builder import GraphBuilder
from src.graph.tracer import PIDTracer
from src.agents.prompts import PID_ANALYSIS_PROMPT, PIPE_IDENTIFICATION_PROMPT

logger = setup_logger(__name__)

# Load environment variables
load_dotenv()


class PIDResult(BaseModel):
    """Structured output for P&ID analysis."""
    pipe: str = Field(..., description="Pipe label")
    from_equipment: str = Field(alias="from", description="Source equipment label")
    to_equipment: str = Field(alias="to", description="Destination equipment label")
    confidence: float = Field(..., description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(None, description="Analysis reasoning")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

    class Config:
        allow_population_by_field_name = True
        json_schema_extra = {
            "example": {
                "pipe": "300-P-310305-NB01-HC",
                "from": "E-3118",
                "to": "Off-page",
                "confidence": 0.95,
                "reasoning": "300-P-310305-NB01-HC connects E-3118 to an off-page connection",
                "processing_time_ms": 2341
            }
        }


class PIDAgent:
    """Main agent for P&ID analysis using LangChain and LangGraph."""

    def __init__(self, model_name: str = None):
        """
        Initialize PID Agent.

        Args:
            model_name: Gemini model name (default: gemini-2.5-flash)
        """
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemini-2.5-flash")
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        # Initialize components
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key,
        )
        self.ocr_extractor = PaddleOCRExtractor()
        self.pipe_detector = PipeDetector()
        self.equipment_detector = EquipmentDetector()
        self.graph_builder = GraphBuilder()
        self.tracer = None

        # Build workflow
        self.workflow = self._build_workflow()
        logger.info(f"Initialized PIDAgent with model: {self.model_name}")

    def _build_workflow(self):
        """
        Build LangGraph workflow.

        Returns:
            Compiled workflow graph
        """
        # Define state schema
        class AgentState(BaseModel):
            image_path: str
            target_pipe: str
            ocr_results: Optional[Dict] = None
            equipment: Optional[list] = None
            pipes: Optional[Dict] = None
            graph: Optional[Dict] = None
            trace_result: Optional[Dict] = None
            result: Optional[Dict] = None
            error: Optional[str] = None

        # Define workflow nodes
        def load_image_node(state: AgentState) -> dict:
            """Load and preprocess image."""
            try:
                logger.info(f"Loading image: {state.image_path}")
                _image = ImagePreprocessor.load_image(state.image_path)
                # Do not store raw image in the Pydantic state model; return no-op update
                return {}
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return {"error": str(e)}

        def ocr_node(state: AgentState) -> dict:
            """Extract text using OCR."""
            try:
                logger.info("Running OCR extraction")
                ocr_results = self.ocr_extractor.extract_text(state.image_path)
                return {"ocr_results": ocr_results}
            except Exception as e:
                logger.error(f"Error in OCR: {e}")
                return {"error": str(e)}

        def equipment_detection_node(state: AgentState) -> dict:
            """Detect equipment labels."""
            try:
                logger.info("Detecting equipment")
                equipment = self.equipment_detector.detect_equipment(state.image_path)
                return {"equipment": equipment}
            except Exception as e:
                logger.error(f"Error in equipment detection: {e}")
                return {"error": str(e)}

        def pipe_detection_node(state: AgentState) -> dict:
            """Detect pipes and pipe labels."""
            try:
                logger.info("Detecting pipes and pipe labels")
                image = ImagePreprocessor.load_image(state.image_path)
                pipes = self.pipe_detector.detect_pipes(image, state.image_path)
                logger.info(f"Detected {len(pipes.get('pipe_labels', []))} pipe labels")
                return {"pipes": pipes}
            except Exception as e:
                logger.error(f"Error in pipe detection: {e}")
                return {"error": str(e)}

        def graph_construction_node(state: AgentState) -> dict:
            """Build graph from detected elements."""
            try:
                logger.info("Building graph")
                if state.equipment and state.pipes:
                    self.graph_builder.build_from_detection(
                        state.equipment,
                        state.pipes["lines"],
                        state.pipes["intersections"],
                        state.pipes.get("pipe_labels", []),
                    )
                    self.tracer = PIDTracer(self.graph_builder.graph)
                    graph_dict = self.graph_builder.to_dict()
                    logger.info(f"Graph: {len(graph_dict['nodes'])} nodes, {len(graph_dict['edges'])} edges")
                    return {"graph": graph_dict}
                return {}
            except Exception as e:
                logger.error(f"Error in graph construction: {e}")
                return {"error": str(e)}

        def tracing_node(state: AgentState) -> dict:
            """Trace pipe connections."""
            try:
                logger.info(f"Tracing pipe: {state.target_pipe}")
                if self.tracer:
                    trace_result = self.tracer.trace_pipe_connections(state.target_pipe)
                else:
                    trace_result = {"error": "Tracer not initialized"}
                return {"trace_result": trace_result}
            except Exception as e:
                logger.error(f"Error in tracing: {e}")
                return {"error": str(e)}

        def output_node(state: AgentState) -> dict:
            """Generate final output."""
            try:
                logger.info("Generating output")

                if state.error:
                    result = {
                        "pipe": state.target_pipe,
                        "from": "Unknown",
                        "to": "Unknown",
                        "confidence": 0.0,
                        "error": state.error,
                    }
                elif state.trace_result:
                    result = {
                        "pipe": state.trace_result.get("pipe", state.target_pipe),
                        "from": state.trace_result.get("from", "Unknown"),
                        "to": state.trace_result.get("to", "Unknown"),
                        "confidence": state.trace_result.get("confidence", 0.5),
                        "reasoning": state.trace_result.get("reasoning"),
                        "source": state.trace_result.get("source", "unknown"),
                    }
                else:
                    result = {
                        "pipe": state.target_pipe,
                        "from": "Unknown",
                        "to": "Unknown",
                        "confidence": 0.0,
                    }

                return {"result": result}
            except Exception as e:
                logger.error(f"Error in output: {e}")
                return {
                    "result": {
                        "pipe": state.target_pipe,
                        "from": "Unknown",
                        "to": "Unknown",
                        "confidence": 0.0,
                        "error": str(e),
                    }
                }

        # Build graph
        workflow = StateGraph(AgentState)

        workflow.add_node("load_image", load_image_node)
        workflow.add_node("ocr", ocr_node)
        workflow.add_node("equipment_detection", equipment_detection_node)
        workflow.add_node("pipe_detection", pipe_detection_node)
        workflow.add_node("graph_construction", graph_construction_node)
        workflow.add_node("tracing", tracing_node)
        workflow.add_node("output", output_node)

        # Set the workflow entry point to the first real node
        workflow.set_entry_point("load_image")

        # Add edges
        # StateGraph injects the START node and routes input to the entry point,
        # so we don't add a START node or edge here.
        workflow.add_edge("load_image", "ocr")
        workflow.add_edge("ocr", "equipment_detection")
        workflow.add_edge("equipment_detection", "pipe_detection")
        workflow.add_edge("pipe_detection", "graph_construction")
        workflow.add_edge("graph_construction", "tracing")
        workflow.add_edge("tracing", "output")
        workflow.add_edge("output", END)

        # Do not connect output back to START (avoids infinite loops)

        return workflow.compile()

    def identify_pipe_from_to(
        self,
        image_path: str,
        target_pipe: str,
    ) -> Dict[str, Any]:
        """
        Identify FROM and TO equipment for a target pipe.

        Args:
            image_path: Path to the P&ID image
            target_pipe: Label of the pipe to identify

        Returns:
            Dictionary with pipe, from, to, and confidence
        """
        start_time = time.time()

        try:
            logger.info(f"Identifying pipe: {target_pipe}")

            # Run workflow
            result = self.workflow.invoke({
                "image_path": image_path,
                "target_pipe": target_pipe,
            })

            processing_time_ms = (time.time() - start_time) * 1000

            output = result.get("result", {})
            output["processing_time_ms"] = processing_time_ms

            logger.info(f"Completed pipe identification in {processing_time_ms:.2f}ms")
            return output

        except Exception as e:
            logger.error(f"Error identifying pipe: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            return {
                "pipe": target_pipe,
                "from": "Unknown",
                "to": "Unknown",
                "confidence": 0.0,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
            }
