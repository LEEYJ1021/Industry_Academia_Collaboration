
# LangGraph Workflow for Adaptive R&D Decision Processes

## Overview

This repository demonstrates a **LangGraph-based workflow** engineered for **dynamic and adaptive decision-making**, specifically within **industry-academia R&D collaboration models**. By integrating **Large Language Models (LLMs)** with structured, data-driven, agentic processes, this framework simulates and optimizes real-world decision-making and R&D planning.

The workflow features **expert-driven approvals**, **iterative feedback loops**, and **conditional routing**, facilitating strategic R&D planning through virtual collaboration scenarios. A visual overview of this dynamic pipeline is presented below:

![Workflow Visualization](LangGraph.png)

---

## Features

- **Data-Driven Persona Generation**  
  Automatically identifies and creates expert personas grounded in real-world patent inventor data.

- **Conditional Routing**  
  Dynamically adjusts workflow pathways based on evaluation scores and human decisions, allowing the system to self-correct.

- **Human-In-The-Loop (HITL) Oversight**  
  Ensures key decisions are aligned with strategic objectives through expert review.

- **Iterative Refinement**  
  Simulated meetings are evaluated, critiqued, and refined until a high-quality R&D plan emerges.

- **Multi-LLM Robustness**  
  Supports GPT-4, Gemini, and Qwen, enabling comparative LLM analysis and operational resilience.

- **Detailed Logging & Monitoring**  
  Captures execution time, token usage, and decisions per node for performance optimization.

---

## Contribution Highlights

- **Workflow Design and Implementation**: Yong-Jae Lee  
- **Visualization Development**: Yong-Jae Lee  
- **Analysis Process Coordination**: Yong-Jae Lee  
- **Results Documentation and Release Planning**: Yong-Jae Lee  

---

## Workflow Structure

Each step in the LangGraph workflow represents a distinct **stateful agentic node** in a decision-making graph.

### Step-by-Step Breakdown

---

### 1. Initialization

- **Node**: `setup_analysis_environment`  
- **Description**:  
  Loads pre-analyzed patent data to identify high-potential technology convergence opportunities (e.g., `G06F16_H04W12`).  
  The user selects one technology pair for simulation.

---

### 2. Expert Persona Engagement

- **Nodes**:  
  - `define_expert_personas`  
  - `human_in_the_loop_expert_approval`  
- **Description**:  
  Generates expert personas based on top inventors in the selected patent domain using LLMs.  
  Human reviewers validate the personas to ensure domain credibility.

---

### 3. Collaboration Strategy and Simulation

- **Nodes**:  
  - `select_convergence_meeting_strategy`  
  - `simulate_convergence_meeting`  
- **Description**:  
  The user selects a strategy (e.g., **Consensus-Driven**, **Greedy-Exploitation**), and a virtual meeting is simulated among expert personas.  
  Strategies are applied to various technology convergence codes:

  **Code Combinations:**
  ```
  B60R21_B60R23
  B60R23_G01S7
  B60R23_Y02T10
  B61L25_G06Q50
  G01C23_G01C5
  G06F16_H04W12
  G07C5_H04M1
  G08G5_H04L20
  H04B7_H04L20
  H04L67_H04R3
  ```

- **Available Strategies**:  
  Defines the strategic direction of the simulated convergence meeting between expert personas. Each strategy affects how the discussion unfolds and how priorities are identified.

  - **Consensus-Driven**  
    Focuses on mediating differences between the two experts to find common ground and reach a consensus-based conclusion.

  - **Exploratory-Brainstorming**  
    Encourages divergent thinking to explore multiple fusion ideas, ultimately identifying the one with the most immediate commercial potential and exploring it in depth.

  - **Greedy-Exploitation**  
    Aims to pinpoint the single most critical academic research field that must be externally introduced to enable successful technology convergence. Ensures that all Action Items reflect this targeted objective.

---

### 4. Feedback and Refinement

- **Nodes**:  
  - `evaluate_convergence_meeting`  
  - `collect_convergence_feedback`  
  - `refine_and_critique`  

- **Description**:  
  Evaluates the simulated meeting outcome using automated scoring and structured feedback generation based on the following criteria:  

  1. **Clarity Score (0.0–10.0):**  
     Measures how clear and understandable the conclusions and action items are.  

  2. **Actionability Score (0.0–10.0):**  
     Assesses how specific and realistically executable the action items are.  

  3. **Alignment Score (0.0–10.0):**  
     Evaluates how well the results align with the original agenda and goals.  

  4. **Qualitative Feedback:**  
     Provides detailed reasons for each score. If a score is low, suggests actionable improvements.  

  5. **Future Metrics Suggestion:**  
     Recommends potential NLP evaluation metrics (e.g., BLEU, ROUGE, BERTScore) to enable automatic and precise quality measurement of meeting logs, including rationale for their selection.  

- **Output:**  
  Produces a structured `MeetingEvaluation` JSON object containing:  
  ```json
{
  "clarity_score": 0.0,
  "actionability_score": 0.0,
  "alignment_score": 0.0,
  "qualitative_feedback": {
    "clarity": "Provide clear explanations for conclusions and action items.",
    "actionability": "Action items should be specific and feasible to implement.",
    "alignment": "Ensure the results directly reflect the original agenda and goals."
  },
  "future_metrics_suggestion": "Consider BLEU for linguistic accuracy, ROUGE for content coverage, and BERTScore for semantic similarity to evaluate meeting logs automatically."
}


---

### 5. Final Synthesis

- **Node**: `synthesize_final_report`  
- **Description**:  
  Integrates all insights, strategies, and meeting logs into a comprehensive R&D plan in Markdown format.

---

## Conditional Routing Logic

The system adapts to context and performance through conditional nodes:

- **Expert Approval (`decide_on_expert_approval`)**:  
  - If expert personas are approved → continue  
  - If not → regenerate personas

- **Feedback Evaluation (`decide_on_convergence_feedback`)**:  
  - If meeting score is sufficient → proceed  
  - If low → refine and repeat

- **Academic Validation (`decide_on_academic_approval`)**:  
  - If selected academic match is appropriate → proceed  
  - Otherwise → retry query and validate again

---

## Code Highlights & How to Run

### Prerequisites

- **Python**: 3.9+  

- **Environment Variables**: Set up the following API keys:
  - `OPENAI_API_KEY`
  - `GOOGLE_API_KEY`
  - `LANGCHAIN_API_KEY`

- **Input Data Files**: Place the following pre-analyzed patent **JSON files in the ./patent_analysis_results_.../ directory**.
These files are also included in the **Releases section**, which contains detailed descriptions:

```
./patent_analysis_results_.../
  ├── Comprehensive_Analysis_Report.json
  └── grouped_by_inventor_applicant.json
```

---

### Running the Workflow

1. **Execute the main script**  
   Run the Python program. It will guide you interactively through:

   - **Model Selection**: Choose LLM backend (GPT-4, Gemini, etc.)
   - **Technology Pair Selection**: Select convergence codes
   - **Expert Persona Approval**: Human-in-the-loop validation
   - **Meeting Strategy Choice**: Define simulation strategy
   - **Feedback Integration**: Review and refine iterations

2. **Final Output.json**  
   The complete R&D proposal will be saved as **Final_Output.json in the Releases section** for your convenience, along with additional documentation.

---
