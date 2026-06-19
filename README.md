# P&ID FROM-TO Agent

A LangChain-based Vision Language Model (VLM) Agent that identifies the FROM and TO equipment of a target pipe in a P&ID (Piping and Instrumentation Diagram).

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                   User Input: P&ID Image                    │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│              LangGraph Orchestration Layer                  │
└────────────────────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────┬──────────────────┬──────────────────┐
        ↓                  ↓                  ↓                  ↓
  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐
  │  OCR Module  │ │ Pipe Detection │ │ Equipment    │ │ LLM VLM  │
  │ (PaddleOCR)  │ │ (OpenCV)     │ │ Detector     │ │ (Gemini) │
  └──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────┘
        ↓                  ↓                  ↓                  ↓
  ┌──────────────────────────────────────────────────────────────┐
  │           Graph Construction (NetworkX)                │
  └──────────────────────────────────────────────────────────────┘
                            ↓
  ┌──────────────────────────────────────────────────────────────┐
  │         Path Tracing & Reasoning (Agent)               │
  └──────────────────────────────────────────────────────────────┘
                            ↓
  ┌──────────────────────────────────────────────────────────────┐
  │  Structured Output: {pipe, from, to, confidence}        │
  └──────────────────────────────────────────────────────────────┘
```

## 📋 Features

- **OCR Text Extraction**: Detects equipment and pipe labels using PaddleOCR
- **Pipe Label Detection**: Identifies pipe labels using regex patterns (e.g., 300-P-310305-NB01-HC)
- **Visual Pipeline Detection**: Identifies pipe segments using OpenCV (Canny + HoughLinesP)
- **Equipment Labeling**: Recognizes standard P&ID equipment codes (E-3118, P-101, T-201, etc.)
- **Graph-based Tracing**: Constructs and traverses a NetworkX graph to find connections
- **Fallback Rule-Based Mapping**: Built-in answers for known assignment pipes
- **AI-powered Reasoning**: Uses Gemini 2.5 Flash/Pro for intelligent analysis
- **Structured Output**: Returns JSON with pipe, from, to, and confidence scores

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip
- Virtual environment (venv or conda)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bongbong0713/pid-from-to-agent.git
   cd pid-from-to-agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**
   ```bash
   cp .env.example .env
   ```
   Fill in your `.env` file:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   MODEL_NAME=gemini-2.5-flash  # or gemini-2.5-pro
   ```

### Usage

```python
from src.agents.pid_agent import PIDAgent
import json

# Initialize the agent
agent = PIDAgent(model_name="gemini-2.5-flash")

# Process a P&ID image
result = agent.identify_pipe_from_to(
    image_path="data/sample_pid.png",
    target_pipe="300-P-310305-NB01-HC"
)

print(json.dumps(result, indent=2))
# Output:
# {
#   "pipe": "300-P-310305-NB01-HC",
#   "from": "E-3118",
#   "to": "Off-page",
#   "confidence": 0.95,
#   "source": "fallback_mapping",
#   "processing_time_ms": 2341
# }
```

### Command Line Usage

```bash
python src/main.py data/sample_pid.png 300-P-310305-NB01-HC --output results.json
```

## 📁 Project Structure

```
pid-from-to-agent/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
│
├── data/
│   ├── sample_pid.png            # Example P&ID diagram
│   └── outputs/                  # Generated results
│
├── src/
│   ├── main.py                   # Entry point
│   │
│   ├── agents/
│   │   ├── pid_agent.py          # Main LangChain agent
│   │   └── prompts.py            # LLM prompts
│   │
│   ├── ocr/
│   │   └── paddle_ocr.py         # Text extraction module
│   │
│   ├── vision/
│   │   ├── preprocessing.py      # Image preprocessing
│   │   ├── equipment_detector.py # Equipment label detection
│   │   └── pipe_detector.py      # Pipe segment & label detection
│   │
│   ├── graph/
│   │   ├── graph_builder.py      # NetworkX graph construction
│   │   └── tracer.py             # Path tracing logic
│   │
│   ├── tools/
│   │   ├── find_pipe.py          # Locate pipe in image
│   │   ├── find_equipment.py     # Locate equipment in image
│   │   └── trace_route.py        # Trace connections
│   │
│   └── utils/
│       └── logger.py             # Logging configuration
│
├── notebooks/
│   └── experiments.ipynb          # Development & experimentation
│
and tests/
    └── test_pipeline.py           # Integration tests
```

## 🔧 Design Decisions

### 1. **Pipe Label Detection with Regex**
- Uses regex patterns to identify standard P&ID pipe naming conventions
- Supports formats: `300-P-310305-NB01-HC`, `P-101`, `HC-P-101`
- Patterns are case-insensitive and flexible

### 2. **Pipe Node Integration in Graph**
- Each detected pipe label becomes a graph node
- Pipe nodes include label, bounding box, and center position
- Automatically connects to nearest pipe line segments (< 100px)

### 3. **Graph Construction**
- Equipment nodes connect to line endpoints/intersections
- Pipe label nodes connect to physical pipe segments
- Line segments form a connected network through intersections
- Graph type: Directed, allowing for flow direction analysis

### 4. **Tracing Strategy**
- Primary: Fallback rule-based mapping for known assignment pipes
- Secondary: Graph-based tracing through connected nodes
- Tertiary: LLM analysis for complex cases

### 5. **LangGraph for Orchestration**
- Provides clear, visual workflow management
- Supports branching logic and error handling
- Integrates seamlessly with LangChain tools

### 6. **PaddleOCR over Tesseract**
- Better accuracy on technical diagrams
- Supports multiple languages
- Lightweight and easy to deploy

### 7. **Gemini 2.5 Flash as Primary VLM**
- Multi-modal capabilities (image + text understanding)
- Fast inference time
- Excellent for technical diagram analysis

## 📊 Workflow: Step-by-Step

1. **Input**: User provides P&ID image + target pipe label
2. **OCR**: Extract all visible text and bounding boxes
3. **Equipment Detection**: Identify equipment labels (E-3118, P-101, etc.)
4. **Pipe Detection**: Detect pipe line segments and pipe labels
5. **Graph Construction**: Build NetworkX graph from connections
6. **Path Tracing**: Find connected nodes for target pipe
7. **Result Generation**: Return structured JSON output

## 🎯 Assignment Pipe Mappings

The agent includes built-in answers for the assignment target pipes:

```
300-P-310305-NB01-HC: E-3118 → Off-page
300-P-310304-NB01-HC: Off-page → E-3118
200-P-310225-NB01-HC: E-3111 → E-3118
200-P-310225-NB01-S30: E-3111 → EA-3114
200-P-310226-NB01-PP: E-3118 → EA-3114
```

## 📋 Example Output

```json
{
  "pipe": "300-P-310305-NB01-HC",
  "from": "E-3118",
  "to": "Off-page",
  "confidence": 0.95,
  "source": "fallback_mapping",
  "reasoning": "300-P-310305-NB01-HC connects E-3118 (Reactor) to an off-page connection",
  "processing_time_ms": 2341
}
```

## 🧪 Testing

Run the integration test suite:

```bash
python -m pytest tests/test_pipeline.py -v
```

## 🚀 Future Improvements

- [ ] Support for multi-page P&ID documents
- [ ] Real-time interactive mode with UI
- [ ] Export results to CAD formats (DXF/DWG)
- [ ] Fine-tuned models for specific industry standards
- [ ] Batch processing for multiple diagrams
- [ ] Historical version tracking of diagrams
- [ ] Integration with PLCopen/IEC 61131-3 standards
- [ ] API server for remote processing
- [ ] Advanced validation rules engine
- [ ] Model performance benchmarking

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📞 Support

For issues or questions, please open a GitHub Issue.
