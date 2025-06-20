# LangGraph Workflow for Adaptive Decision Processes

## Overview

This repository demonstrates a **LangGraph-based workflow** designed for dynamic and adaptive decision-making processes. The framework has been successfully applied in **industry-academia collaboration models**, simulating and optimizing real-world decision-making and R&D planning.

The workflow integrates expert-driven approvals, iterative refinements, and conditional routing to optimize outcomes, enabling efficient technology convergence and innovation in collaborative environments. A visual representation of the workflow, generated using **Mermaid**, is provided in **LangGraph.png**.

Through iterative feedback loops, conditional routing, and expert evaluations, this system models virtual collaboration scenarios that enhance decision-making and R&D strategy development.

![Workflow Visualization](LangGraph.png)

---

## Features

- **Conditional Routing**: Dynamically adjusts workflow pathways based on real-time decisions.
- **Human-In-The-Loop Oversight**: Incorporates critical expert reviews at key decision points.
- **Iterative Refinement**: Integrates feedback loops for continuous process improvement.
- **Industry-Academia Simulation**: Models virtual collaboration scenarios for R&D planning.
- **Visualization**: Provides a clear and interactive representation of workflow structure.
---

### Contribution Highlights
- Workflow Design and Implementation: **Yong-Jae Lee**
- Visualization Development: **Yong-Jae Lee**
- Analysis Process Coordination: **Yong-Jae Lee**
- Results Documentation and Release Planning: **Yong-Jae Lee**

---

## Workflow Structure

### Step-by-Step Breakdown

1. **Initialization**  
   - **Node**: `setup_analysis_environment`  
   Prepares the environment by loading initial data, setting up prerequisites, and configuring analysis parameters.

2. **Expert Persona Engagement**  
   - **Nodes**:  
     - `define_expert_personas`  
     - `human_in_the_loop_expert_approval`  
   Incorporates domain experts into the decision-making process, enabling specialized insights and human-in-the-loop approvals.

3. **Collaboration Strategy and Simulation**  
   - **Nodes**:  
     - `select_convergence_meeting_strategy`  
     - `simulate_convergence_meeting`  
   Plans and tests collaboration strategies between industry and academic experts through virtual simulation.  
   The strategy selection cycles through specific technology code combinations with the following convergence meeting strategies:  
   - **Code Combinations & Strategies**  
     - `B60R21_B60R23`  
     - `B60R23_G01S7`  
     - `B60R23_Y02T10`  
     - `B61L25_G06Q50`  
     - `G01C23_G01C5`  
     - `G06F16_H04W12`  
     - `G07C5_H04M1`  
     - `G08G5_H04L20`  
     - `H04B7_H04L20`  
     - `H04L67_H04R3`  

   Each is explored sequentially using the strategy set:  
   - **Consensus-Driven**  
   - **Exploratory-Brainstorming**  
   - **Greedy-Exploitation**

4. **Feedback and Refinement**  
   - **Nodes**:  
     - `evaluate_convergence_meeting`  
     - `collect_convergence_feedback`  
     - `refine_and_critique`  
   Collects and integrates expert feedback iteratively to optimize collaboration outcomes and refine strategies.

5. **Final Synthesis**  
   - **Node**: `synthesize_final_report`  
   Produces a comprehensive R&D proposal or actionable report in Markdown format, summarizing results and recommendations.


### Conditional Routing Logic
- **Expert Approval**:
  - Routes to `continue` or `regenerate` based on expert evaluation.
- **Feedback Evaluation**:
  - Directs workflow to either `proceed` or `revise` based on feedback.
- **Query and Academic Validation**:
  - Manages iterative adjustments until conditions are met.

---
