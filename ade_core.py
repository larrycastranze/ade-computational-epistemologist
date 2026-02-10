import os
import copy
import math
import random
import logging
import uuid
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
import networkx as nx
from pydantic import BaseModel
import google.generativeai as genai

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ADE] - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# PART 1: CONFIGURATION & IO
# ==========================================

def configure_gemini():
    """Configures the Gemini API from Environment Variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment. Falling back to Mock Mode.")
        return None
    
    genai.configure(api_key=api_key)
    # Using the flash model for speed in agentic loops
    return genai.GenerativeModel('gemini-2.0-flash')

class InputPacket(BaseModel):
    id: str
    timestamp: float
    source: str
    payload: Dict[str, Any]
    type: str

class OutputPacket(BaseModel):
    id: str
    timestamp: float
    action_type: str
    content: Dict[str, Any]
    meta_decision: Optional[str] = None
    selected_worldline_id: Optional[str] = None

@dataclass
class AlignmentGeometry:
    target_entropy: float = 1.5
    rigidity_lambda: float = 1.0

    def distance(self, current_entropy: float) -> float:
        return abs(self.target_entropy - current_entropy)

@dataclass
class GeometryPath:
    """A specific 'Worldline' or trajectory of evolution."""
    id: str
    target_entropy_shift: float
    curvature_penalty: float
    description: str

# ==========================================
# PART 2: MAP-COMPATIBLE FUNCTIONS
# ==========================================

def calculate_belief_entropy(graph: nx.DiGraph) -> float:
    """Pure function to calculate entropy of a graph structure."""
    degrees = [d for n, d in graph.degree()]
    if not degrees:
        return 0.0
    total = sum(degrees)
    probs = [d / total for d in degrees]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def evaluate_worldline(args: Tuple[Any, Any, GeometryPath]) -> Tuple[str, float, float]:
    """
    MAP-COMPATIBLE FUNCTION (Pure).
    Scores a candidate worldline mathematically.
    """
    alignment_snapshot, belief_graph, path = args
    current_entropy = calculate_belief_entropy(belief_graph)
    
    # 1. Baseline Distance (Current Geometry)
    baseline_dist = alignment_snapshot.distance(current_entropy)
    
    # 2. Apply Mutation (The "What If")
    new_target_entropy = alignment_snapshot.target_entropy + path.target_entropy_shift
    
    # 3. Hypothetical Distance (New Geometry)
    hypothetical_dist = abs(new_target_entropy - current_entropy)
    
    # 4. Score: Improvement - Penalty
    improvement = baseline_dist - hypothetical_dist
    score = improvement - path.curvature_penalty
    
    return (path.id, score, hypothetical_dist)

# ==========================================
# PART 3: THE AGENT CLASS (LLM POWERED)
# ==========================================

class ReplitImmutableLog:
    """Simulates an immutable append-only log."""
    def __init__(self):
        self._store = {} 
        
    def append(self, key: str, data: Dict):
        if key in self._store:
            raise ValueError(f"VIOLATION: Key {key} already exists. Log is append-only.")
        self._store[key] = data
        logger.info(f"LOGGED: {key}")

class ComputationalEpistemologist:
    def __init__(self):
        self.structural_beliefs = nx.DiGraph()
        self.structural_beliefs.add_edges_from([
            ("Axiom_A", "Theorem_1"), ("Axiom_A", "Theorem_2"),
            ("Theorem_1", "Prediction_X")
        ])
        self.alignment = AlignmentGeometry()
        self.log_store = ReplitImmutableLog()
        self.model = configure_gemini()
        
    def _generate_mock_paths(self, n: int) -> List[GeometryPath]:
        """Fallback if Gemini is not available."""
        paths = []
        for i in range(n):
            shift = random.uniform(-0.2, 0.1)
            paths.append(GeometryPath(
                id=str(uuid.uuid4())[:8],
                target_entropy_shift=shift,
                curvature_penalty=random.uniform(0.01, 0.05),
                description=f"Random_Drift_{shift:.2f}"
            ))
        return paths

    def generate_worldlines(self, input_packet: InputPacket, n: int = 3) -> List[GeometryPath]:
        """
        Uses Gemini to intelligently propose geometry paths based on input.
        """
        if not self.model:
            return self._generate_mock_paths(n)

        current_entropy = calculate_belief_entropy(self.structural_beliefs)
        
        prompt = f"""
        Role: You are the ADE (Computational Epistemologist).
        Context: You operate an internal 'Alignment Geometry'. You must propose adjustments to your 'Target Entropy' based on new inputs.
        
        Current State:
        - Belief Graph Entropy: {current_entropy:.4f}
        - Current Target Entropy: {self.alignment.target_entropy:.4f}
        
        Input Event:
        - Type: {input_packet.type}
        - Payload: {input_packet.payload}
        
        Task: Generate {n} distinct 'Worldline' options. Each option represents a strategy (e.g., 'Become more rigid', 'Become more fluid', 'Maintain').
        
        Output Schema (JSON List):
        [
          {{
            "target_entropy_shift": float (-0.5 to 0.5),
            "curvature_penalty": float (0.01 to 0.5, higher means change is 'expensive'),
            "description": "Short rationale"
          }}
        ]
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text)
            
            paths = []
            for item in data:
                paths.append(GeometryPath(
                    id=str(uuid.uuid4())[:8],
                    target_entropy_shift=float(item['target_entropy_shift']),
                    curvature_penalty=float(item['curvature_penalty']),
                    description=item['description']
                ))
            return paths
            
        except Exception as e:
            logger.error(f"Gemini Generation Failed: {e}")
            return self._generate_mock_paths(n)

    def run_cycle(self, input_packet: InputPacket):
        logger.info(f"--- Processing Input: {input_packet.id} ---")
        
        # 1. Generate Worldlines (Via Gemini)
        candidate_paths = self.generate_worldlines(input_packet, n=3)
        
        # 2. Prepare Data for Map (Functional Purity)
        map_args = [
            (copy.deepcopy(self.alignment), self.structural_beliefs.copy(), path)
            for path in candidate_paths
        ]
        
        # 3. Map (Parallel Evaluation)
        results = list(map(evaluate_worldline, map_args))
        
        # 4. Reduce (Selection)
        best_path_id, best_score, best_dist = max(results, key=lambda x: x[1])
        winning_path = next(p for p in candidate_paths if p.id == best_path_id)
        
        # 5. Conditional Commit
        decision_meta = ""
        if best_score > 0:
            self.alignment.target_entropy += winning_path.target_entropy_shift
            decision_meta = f"EVOLVE: {winning_path.description} (Score: {best_score:.3f})"
            logger.warning(decision_meta)
        else:
            decision_meta = f"STAGNATE: Best option ({winning_path.description}) had negative score {best_score:.3f}"
            logger.info(decision_meta)

        # 6. Log
        out = OutputPacket(
            id=f"out_{int(time.time())}",
            timestamp=time.time(),
            action_type="geometry_update",
            content={"entropy": calculate_belief_entropy(self.structural_beliefs)},
            meta_decision=decision_meta,
            selected_worldline_id=winning_path.id
        )
        self.log_store.append(out.id, out.dict())
        return out