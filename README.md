
# LangGraph Workflow for Adaptive R&D Decision Processes

## 1. Overview
This repository demonstrates a **LangGraph-based workflow** engineered for **dynamic and adaptive decision-making**, specifically within **industry-academia R&D collaboration models**.  

By integrating **Large Language Models (LLMs)** with **structured, data-driven, agentic processes**, this framework **simulates and optimizes real-world decision-making and R&D planning**.

The workflow features:
- **Expert-driven approvals**
- **Iterative feedback loops**
- **Conditional routing**

These elements together facilitate **strategic R&D planning through virtual collaboration scenarios**.

---

## Visual Workflow Overview

![Workflow Visualization](LangGraph.png)

---

## 2. Features

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

## 3. Contribution Highlights

- **Workflow Design and Implementation**: Yong-Jae Lee  
- **Visualization Development**: Yong-Jae Lee  
- **Analysis Process Coordination**: Yong-Jae Lee  
- **Results Documentation and Release Planning**: Yong-Jae Lee  

---

## 4. Workflow Structure

Each step in the LangGraph workflow represents a distinct **stateful agentic node** in a decision-making graph.

### Step-by-Step Breakdown

---

### 4.1. Initialization

- **Node**: `setup_analysis_environment`  
- **Description**:  
  Loads pre-analyzed patent data to identify high-potential technology convergence opportunities (e.g., `G06F16_H04W12`).  
  The user selects one technology pair for simulation.

---

### 4.2. Expert Persona Engagement

- **Nodes**:  
  - `define_expert_personas`  
  - `human_in_the_loop_expert_approval`  
- **Description**:  
  Generates expert personas based on top inventors in the selected patent domain using LLMs.  
  Human reviewers validate the personas to ensure domain credibility.

---

### 4.3. Collaboration Strategy and Simulation

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

### 4.4. Feedback and Refinement

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

### 4.5. Final Synthesis

- **Node**: `synthesize_final_report`  
- **Description**:  
  Integrates all insights, strategies, and meeting logs into a comprehensive R&D plan in Markdown format.

---

## 5. Conditional Routing Logic

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

# 6. Results and Performance Analysis  

## 6.1. Experimental Evaluation and Performance Analysis  

### 6.1.1. Experimental Design and Setup  
- **30 simulations** conducted using 10 high-potential technology convergence pairs.  
- **3 meeting facilitation strategies** tested:  
  - **Consensus-Driven**: focuses on agreement.  
  - **Greedy-Exploitation**: prioritizes short-term high-impact actions.  
  - **Exploratory-Brainstorming**: fosters diverse ideas without immediate consensus.  

### 6.1.2. Simulation Environment and Implementation  
- **Model**: Qwen LLM (local HPC deployment).  
- **Hardware**: 8-core Intel Xeon, 4× V100 GPUs, 128GB RAM, 2TB NVMe SSD.  
- **Software**: Python 3.8, PyTorch 1.12, Sentence-Transformers 2.2.2, Neo4j 4.4.11.  
- **Execution setup**:  
  - Batch size: 64  
  - Parallel threads: 16 workers + 4 I/O  
  - 5 random seeds for robustness  
- **Metrics tracked**: execution time per node, token usage, and variance (<3%).  

### 6.1.3. Token Usage–Execution Time Correlation  
- **Pearson correlation**: **r = 0.899, p = 0.0059**.  
- Strong positive correlation → higher token usage → longer execution time.  
- **Bottlenecks**:  
  - `synthesize_final_report_node`  
  - `execute_arxiv_search_and_create_persona_node`  
- **Efficient outlier**: `generate_arxiv_query_node`.  

### 6.1.4. Node-Level Performance Analysis  
**Efficiency metrics:**  
- **Token Efficiency** (output/input ratio): High = `synthesize_final_report_node (0.88)`; Low = `generate_arxiv_query_node (0.09)`.  
- **Processing Efficiency** (time per token): Fastest = `generate_arxiv_query_node (1.85 ms/token)`; Slowest = `execute_arxiv_search_and_create_persona_node (12.36 ms/token)`.  
- **Output Productivity** (tokens/sec): Highest = `simulate_industry_academia_collaboration_node (72.78)`; Lowest = `execute_arxiv_search_and_create_persona_node (10.19)`.  

**Stability (CV of execution time):**  
- **High variability**: `execute_arxiv_search_and_create_persona_node`, `collect_convergence_feedback_node`.  
- **Low variability**: `generate_arxiv_query_node`, `synthesize_final_report_node`.  

**Resource bottlenecks:**  
- `synthesize_final_report_node`: **34.6% tokens, 29.3% time**.  
- Top 3 nodes consume **61.5% total execution time**.  
- Optimizing top 3 nodes by 20% → **12.3% overall improvement**.

### 6.1.5. Optimization Strategy and Recommendations  
1. **Prioritize bottlenecks**  
   - Optimize `synthesize_final_report_node` (parallelization, incremental generation).  
   - Refactor `execute_arxiv_search_and_create_persona_node` (caching, async processing).  
   - Improve `evaluate_meeting_outcome_node`.  

2. **Stabilize high-variability nodes**  
   - Introduce **retry logic**, **timeouts**, **asynchronous processing**.  

3. **Enhance token efficiency**  
   - **Prompt engineering** for `generate_arxiv_query_node` (increase info density).  
   - **Processing logic optimization** for `synthesize_final_report_node`.  

**Phased plan:**  
- **Short-term**: stabilize variability.  
- **Medium-term**: optimize top bottlenecks.  
- **Long-term**: improve token efficiency framework-wide.  

---

## 7. Code Highlights & How to Run

### 7.1. Prerequisites

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

### 7.2. Running the Workflow

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

# Appendix

## Appendix A: Performance Evaluation Details
This appendix provides detailed statistical and visual analyses of computational performance across the **adaptive multi-LLM agent simulation framework**.  
All metrics were collected from **30 end-to-end simulations** spanning **ten technology convergence pairs** and **three facilitation strategies**.  
Each simulation instance was repeated **five times under different random seeds** to ensure statistical robustness.  

The evaluation covered two primary dimensions:  
1. **Node-level execution time**  
2. **Token usage** (input, output, and total)  

This enabled capturing both **processing efficiency** and **computational load**.

---

### TABLE A.1. Node-Level Execution Time Distribution

| NODE NAME | N | MIN | Q1 | MEDIAN | MEAN | Q3 | MAX |
|-----------|---|-----|----|--------|------|----|-----|
| SYNTHESIZE_FINAL_REPORT_NODE | 30 | 23.12 | 25.09 | 25.64 | 26.41 | 27.38 | 32.60 |
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 30 | 4.23 | 8.88 | 17.81 | 21.85 | 23.55 | 72.53 |
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 30 | 3.87 | 4.33 | 4.74 | 4.72 | 5.04 | 5.49 |
| DEFINE_EXPERT_PERSONAS_NODE | 30 | 3.34 | 4.00 | 4.22 | 4.24 | 4.55 | 5.03 |
| SIMULATE_CONVERGENCE_MEETING_NODE | 30 | 3.33 | 3.88 | 4.16 | 4.22 | 4.69 | 5.14 |
| COLLECT_CONVERGENCE_FEEDBACK_NODE | 30 | 1.54 | 2.10 | 3.01 | 3.95 | 3.96 | 23.86 |
| HUMAN_IN_THE_LOOP_QUERY_APPROVAL_NODE | 30 | 2.03 | 2.62 | 3.15 | 3.63 | 3.92 | 11.22 |
| EVALUATE_MEETING_OUTCOME_NODE | 60 | 2.81 | 3.40 | 3.57 | 3.55 | 3.72 | 4.20 |
| HUMAN_IN_THE_LOOP_ACADEMIC_APPROVAL_NODE | 30 | 1.77 | 2.29 | 2.65 | 2.83 | 3.09 | 4.95 |
| HUMAN_IN_THE_LOOP_EXPERT_APPROVAL_NODE | 30 | 1.65 | 2.00 | 2.33 | 2.67 | 3.24 | 5.20 |
| SETUP_ANALYSIS_ENVIRONMENT_NODE | 30 | 1.29 | 1.79 | 2.35 | 2.56 | 3.09 | 5.25 |
| SELECT_MEETING_STRATEGY_NODE | 60 | 1.55 | 1.91 | 2.22 | 2.33 | 2.64 | 3.97 |
| GENERATE_ARXIV_QUERY_NODE | 30 | 1.03 | 1.12 | 1.17 | 1.17 | 1.20 | 1.33 |

---

### TABLE A.2. Node-Level Token Usage Distribution

| TOKEN TYPE | NODE NAME | N | MIN | Q1 | MEDIAN | MEAN | Q3 | MAX |
|------------|-----------|---|-----|----|--------|------|----|-----|
| INPUT_TOKENS | EVALUATE_MEETING_OUTCOME_NODE | 60 | 715 | 769.75 | 807.5 | 803.9 | 836.5 | 890 |
| INPUT_TOKENS | DEFINE_EXPERT_PERSONAS_NODE | 30 | 749 | 751 | 756.5 | 758.5 | 769 | 770 |
| INPUT_TOKENS | EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 30 | 680 | 1529 | 1607 | 1545.4 | 1616 | 1755 |
| INPUT_TOKENS | GENERATE_ARXIV_QUERY_NODE | 30 | 573 | 578 | 579 | 579.9 | 582.75 | 587 |
| INPUT_TOKENS | SIMULATE_CONVERGENCE_MEETING_NODE | 30 | 570 | 578.5 | 584 | 583.43 | 589 | 592 |
| INPUT_TOKENS | SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 30 | 586 | 593 | 595 | 595.03 | 597.75 | 601 |
| INPUT_TOKENS | SYNTHESIZE_FINAL_REPORT_NODE | 30 | 1893 | 2020.25 | 2088 | 2071.83 | 2137.25 | 2238 |
| OUTPUT_TOKENS | EVALUATE_MEETING_OUTCOME_NODE | 60 | 176 | 224.75 | 239 | 238.48 | 252.25 | 293 |
| OUTPUT_TOKENS | DEFINE_EXPERT_PERSONAS_NODE | 30 | 231 | 277 | 295 | 295.33 | 319.25 | 359 |
| OUTPUT_TOKENS | EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 30 | 174 | 199.75 | 226 | 222.57 | 243.5 | 265 |
| OUTPUT_TOKENS | GENERATE_ARXIV_QUERY_NODE | 30 | 43 | 50 | 53.5 | 53.07 | 56.75 | 65 |
| OUTPUT_TOKENS | SIMULATE_CONVERGENCE_MEETING_NODE | 30 | 231 | 276.5 | 297.5 | 303.2 | 342.5 | 377 |
| OUTPUT_TOKENS | SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 30 | 275 | 312.5 | 344.5 | 343.53 | 368 | 404 |
| OUTPUT_TOKENS | SYNTHESIZE_FINAL_REPORT_NODE | 30 | 1617 | 1733.5 | 1779 | 1826.83 | 1901.75 | 2255 |
| TOTAL_TOKENS | EVALUATE_MEETING_OUTCOME_NODE | 60 | 931 | 1008.75 | 1050.5 | 1042.38 | 1084.5 | 1127 |
| TOTAL_TOKENS | DEFINE_EXPERT_PERSONAS_NODE | 30 | 980 | 1033 | 1058 | 1053.83 | 1074.5 | 1128 |
| TOTAL_TOKENS | EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 30 | 928 | 1725.75 | 1822.5 | 1767.97 | 1861.5 | 1995 |
| TOTAL_TOKENS | GENERATE_ARXIV_QUERY_NODE | 30 | 624 | 627.5 | 633 | 632.97 | 637.5 | 648 |
| TOTAL_TOKENS | SIMULATE_CONVERGENCE_MEETING_NODE | 30 | 805 | 856.25 | 878.5 | 886.63 | 925.5 | 961 |
| TOTAL_TOKENS | SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 30 | 868 | 909 | 940 | 938.57 | 964.75 | 1004 |
| TOTAL_TOKENS | SYNTHESIZE_FINAL_REPORT_NODE | 30 | 3560 | 3805.75 | 3860 | 3898.67 | 4050 | 4246 |

---

## Appendix B: Detailed Performance Data
This appendix contains supplementary tables that provide **granular performance data**, supporting the analysis in **Section 6.1.4** and **Section 6.1.5**.

---

### TABLE B.1. Node-Level Token Efficiency (Output/Input Ratio)

| NODE NAME | MEAN INPUT | MEAN OUTPUT | OUTPUT/INPUT RATIO |
|-----------|------------|-------------|---------------------|
| SYNTHESIZE_FINAL_REPORT_NODE | 2071.83 | 1826.83 | 0.88 |
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 595.03 | 343.53 | 0.58 |
| SIMULATE_CONVERGENCE_MEETING_NODE | 583.43 | 303.20 | 0.52 |
| DEFINE_EXPERT_PERSONAS_NODE | 758.50 | 295.33 | 0.39 |
| EVALUATE_MEETING_OUTCOME_NODE | 803.90 | 238.48 | 0.30 |
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 1545.40 | 222.57 | 0.14 |
| GENERATE_ARXIV_QUERY_NODE | 579.90 | 53.07 | 0.09 |

---

### TABLE B.2. Node-Level Processing Efficiency (Time per Token)

| NODE NAME | MEAN TIME (S) | MEAN TOTAL TOKENS | TIME PER TOKEN (MS) | TOKENS PER SECOND |
|-----------|---------------|--------------------|---------------------|-------------------|
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 21.85 | 1767.97 | 12.36 | 80.91 |
| SYNTHESIZE_FINAL_REPORT_NODE | 26.41 | 3898.67 | 6.77 | 147.62 |
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 4.72 | 938.57 | 5.03 | 198.85 |
| SIMULATE_CONVERGENCE_MEETING_NODE | 4.22 | 886.63 | 4.76 | 210.10 |
| DEFINE_EXPERT_PERSONAS_NODE | 4.24 | 1053.83 | 4.02 | 248.55 |
| EVALUATE_MEETING_OUTCOME_NODE | 3.55 | 1042.38 | 3.41 | 293.63 |
| GENERATE_ARXIV_QUERY_NODE | 1.17 | 632.97 | 1.85 | 541.00 |

---

### TABLE B.3. Node-Level Output Productivity

| NODE NAME | MEAN OUTPUT | MEAN TIME (S) | OUTPUT PRODUCTIVITY (TOKENS/S) |
|-----------|-------------|---------------|--------------------------------|
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 343.53 | 4.72 | 72.78 |
| SIMULATE_CONVERGENCE_MEETING_NODE | 303.20 | 4.22 | 71.85 |
| DEFINE_EXPERT_PERSONAS_NODE | 295.33 | 4.24 | 69.65 |
| SYNTHESIZE_FINAL_REPORT_NODE | 1826.83 | 26.41 | 69.17 |
| EVALUATE_MEETING_OUTCOME_NODE | 238.48 | 3.55 | 67.18 |
| GENERATE_ARXIV_QUERY_NODE | 53.07 | 1.17 | 45.36 |
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 222.57 | 21.85 | 10.19 |

---

### TABLE B.4. Node-Level Variability in Execution Time and Token Usage

| NODE NAME | TIME IQR | TIME CV | TOKEN IQR | TOKEN CV |
|-----------|----------|---------|-----------|----------|
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 14.67 | 0.82 | 135.75 | 0.07 |
| SIMULATE_CONVERGENCE_MEETING_NODE | 0.81 | 0.19 | 69.25 | 0.08 |
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 0.71 | 0.15 | 55.75 | 0.06 |
| DEFINE_EXPERT_PERSONAS_NODE | 0.55 | 0.13 | 41.50 | 0.04 |
| EVALUATE_MEETING_OUTCOME_NODE | 0.32 | 0.09 | 75.75 | 0.07 |
| SYNTHESIZE_FINAL_REPORT_NODE | 2.29 | 0.09 | 244.25 | 0.06 |
| GENERATE_ARXIV_QUERY_NODE | 0.08 | 0.07 | 10.00 | 0.02 |

---

### TABLE B.5. Node-Level Resource Consumption Share

| NODE NAME | TOTAL TOKENS USED | TOKEN SHARE (%) | TOTAL TIME USED | TIME SHARE (%) |
|-----------|-------------------|-----------------|-----------------|----------------|
| SYNTHESIZE_FINAL_REPORT_NODE | 116960.01 | 34.61 | 792.30 | 29.34 |
| EVALUATE_MEETING_OUTCOME_NODE | 62542.98 | 18.51 | 213.00 | 7.89 |
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 53039.01 | 15.70 | 655.50 | 24.28 |
| DEFINE_EXPERT_PERSONAS_NODE | 31614.99 | 9.36 | 127.20 | 4.71 |
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 28157.00 | 8.33 | 141.60 | 5.24 |
| SIMULATE_CONVERGENCE_MEETING_NODE | 26599.00 | 7.87 | 126.60 | 4.69 |
| GENERATE_ARXIV_QUERY_NODE | 18989.00 | 5.62 | 35.10 | 1.30 |

---

### TABLE B.6. Node Efficiency vs. Complexity Analysis

| NODE NAME | MEAN TOTAL TOKENS | TOKEN CV | COMPLEXITY SCORE | TIME PER TOKEN (MS) | EFFICIENCY SCORE |
|-----------|-------------------|----------|------------------|---------------------|------------------|
| SYNTHESIZE_FINAL_REPORT_NODE | 3898.67 | 0.063 | 2.410 | 6.77 | -0.387 |
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA_NODE | 1767.97 | 0.074 | 1.020 | 12.36 | -2.030 |
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION_NODE | 938.57 | 0.059 | -0.385 | 5.03 | 0.126 |
| DEFINE_EXPERT_PERSONAS_NODE | 1053.83 | 0.039 | -1.171 | 4.02 | 0.422 |
| SIMULATE_CONVERGENCE_MEETING_NODE | 886.63 | 0.079 | 0.433 | 4.76 | 0.205 |
| EVALUATE_MEETING_OUTCOME_NODE | 1042.38 | 0.072 | 0.273 | 3.41 | 0.603 |
| GENERATE_ARXIV_QUERY_NODE | 632.97 | 0.016 | -2.580 | 1.85 | 1.062 |

---
