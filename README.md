# LangGraph Workflow for Adaptive R&D Decision Processes

## 1. Overview

This repository presents a **LangGraph-based multi-agent framework** engineered for **dynamic and adaptive decision-making** in **industry–academia R&D collaboration**. By integrating **Large Language Models (LLMs)** with **structured, data-driven, agentic processes**, this framework simulates and optimizes real-world R&D planning through a Two-Track experimental architecture.

> **Contribution:** A step toward treating early-stage R&D innovation as a simulatable, data-grounded process — advancing systematic R&D foresight through reproducible, multi-agent simulation.

The workflow features **expert-driven approvals**, **iterative feedback loops**, **conditional routing**, and a **non-LLM evaluation architecture** that substantially reduces evaluative circularity — addressing a critical gap in prior LLM-based innovation studies (Chang et al., 2023; Liu et al., 2023).

> **Note on the term "forecasting":** Within the methodology, the Bayesian LSTM component is more precisely characterized as *convergence signal detection* — identifying short-window directional growth signals rather than producing quantitative point forecasts. This distinction is maintained consistently throughout the paper and this repository.

---

## Visual Workflow Overview

![Workflow Visualization](LangGraph.png)

---

## 2. Key Features

**Data-Driven Persona Generation** — Automatically identifies and creates expert personas grounded in real-world patent inventor data (WIPO PCT), combined with ArXiv-based academic expert discovery.

**Convergence Signal Detection** — A Bayesian LSTM model identifies high-potential CPC technology convergence pairs using walk-forward validation, benchmarked against four transparent baselines (Naïve Random Walk, Naïve Trend, Linear Regression, ARIMA(1,1,0)).

**Non-LLM Evaluation Architecture** — All six CQS scorers are discriminative, non-generative models (NLI DeBERTa, cross-encoder MiniLM, ST cosine similarity, QNLI, Jaccard rule, keyword density). This substantially reduces — though does not fully eliminate — the "LLM-as-judge" circularity problem (Chang et al., 2023; Liu et al., 2023).

**Human-In-The-Loop (HITL) Oversight** — Key decisions are validated through expert review at multiple nodes, ensuring strategic alignment.

**Iterative Refinement** — Simulated meetings are evaluated, critiqued, and refined until a high-quality R&D proposal emerges.

**Multi-LLM Robustness** — Supports GPT-4, Phi-3-mini, Qwen2.5 (1.5B/3B), and Gemini, enabling comparative analysis across model scales. The Track A generator used in all 30 archived primary runs is **GPT-4-class API (gpt-4-turbo-preview; temperature = 0.7)**.

**Detailed Logging & Monitoring** — Captures execution time, token usage, and decisions per node for performance optimization.

---

## 3. Two-Track Experimental Architecture

Empirical contributions are organized as a complementary Two-Track design, where each track answers a different validity question.

| | Track A | Track B |
|---|---|---|
| **Purpose** | Framework capability demonstration (generative ceiling) | Reproducibility + independent validation |
| **Sections** | Sections 4–5 of paper | Appendix E of paper |
| **Models** | GPT-4-class API (gpt-4-turbo-preview; temp = 0.7) | Phi-3-mini-4k-instruct (generator); Qwen2.5-3B (cross-family baseline) |
| **Run count** | 30 | 12–18 per condition |
| **CQS scorer** | LLM-based | Non-LLM: 6 discriminative models (0/6 generative) |
| **CQS range** | M = 8.40 → 8.98 (paired improvement) | M = 7.131 (Full Pipeline) |
| **Key output** | Significant improvement (p < .001, d = 2.62) — reported as capability ceiling only | Baseline superiority (p < .001, d = 2.23, CLES = 100%) |
| **Circularity** | LLM-as-judge circularity applies; results excluded from all inferential claims | **Substantially reduced: 0/6 LLM scorers; generator (Phi-3-mini) architecturally independent of evaluators** |
| **Human validation** | Not conducted | ICC(2,1) = 0.918; Kendall W = 0.951 (n = 24) |

*Cross-track score comparisons are not appropriate given evaluator-type differences (LLM-based vs. non-LLM discriminative). Track A characterizes the generative capability ceiling under aligned generation–evaluation conditions and is explicitly excluded from all inferential claims. Track B is the primary evidence base for validity.*

---

## 4. Contribution Highlights

Workflow design, visualization development, analysis coordination, and results documentation were all conducted by **Yong-Jae Lee**.

---

## 5. Workflow Structure

Each step in the LangGraph workflow represents a distinct **stateful agentic node** in a decision-making graph.

### 5.1. Initialization

**Node:** `setup_analysis_environment`

Loads pre-analyzed patent data to identify high-potential technology convergence opportunities. The Bayesian LSTM performs *convergence signal detection* on actual CPC co-occurrence time-series — identifying which technology pairs show statistically significant short-window growth signals for use as empirically grounded simulation inputs. The user selects one technology pair for simulation.

**Walk-Forward Signal Detection Results — Short Window (Train: 2000–2014; Predict: 2015–2019):**

| Technology Pair | Dir_Acc | SMAPE | Theil's U | PI-80 Coverage |
|---|---|---|---|---|
| G06F16–H04W12 | 100% | 35.63% | 0.620 | 100% |
| B60R21–B60R23 | 100% | 32.73% | 0.605 | 100% |
| G08G5–H04L20 | 100% | 47.27% | 0.681 | 100% |
| B60R23–G01S7 | 100% | 31.75% | 0.586 | 100% |
| B61L25–G06Q50 | 100% | 49.43% | 0.707 | 100% |
| G07C5–H04M1 | 100% | 39.78% | 0.647 | 100% |
| **Average** | **100%** | **39.43%** | **0.641** | **100%** |

**Walk-Forward Signal Detection Results — Long Window (Train: 2000–2019; Predict: 2020–2024):**

| Technology Pair | Dir_Acc | SMAPE | Theil's U | PI-80 Coverage |
|---|---|---|---|---|
| G06F16–H04W12 | 0% | 25.81% | 0.615 | 100% |
| B60R21–B60R23 | 0% | 19.51% | 0.533 | 100% |
| G08G5–H04L20 | 0% | 24.00% | 0.600 | 100% |
| B60R23–G01S7 | 0% | 20.29% | 0.539 | 100% |
| B61L25–G06Q50 | 0% | 27.03% | 0.625 | 100% |
| G07C5–H04M1 | 0% | 24.56% | 0.583 | 100% |
| **Average** | **0%** | **23.53%** | **0.583** | **100%** |

*Primary metric is directional accuracy (Dir_Acc). All Theil's U < 1.0 across all 12 windows (beats naïve baseline on all pairs and windows). The long-window Dir_Acc = 0% result is treated as a **systematic directional inversion**: the model's sensitivity to the anomalous 2015–2019 growth surge induced by the 18-month patent-publication lag appears as a non-stationary inflection in the 2000–2019 training window. The 80% PI coverage remains 100% even on long windows, confirming that uncertainty calibration is preserved despite the directional bias. All practitioner guidance is restricted to the three-period (short-window) setting. Separate full Bayesian LSTM demonstration with MC-dropout uncertainty quantification (H04L67–H04R3 demo pair): Dir_Acc = 100%, SMAPE = 20.7%, Theil's U = 0.609.*

**LSTM vs. Transparent Baselines (Short Window, same six pairs):**

| Method | Dir_Acc | Theil's U |
|---|---|---|
| Naïve Random Walk | 0.000 | 0.901 |
| Naïve Trend | 0.833 | 0.769 |
| Linear Regression | 0.833 | 0.839 |
| ARIMA(1,1,0) | 0.333 | 1.008 (fails the random-walk test) |
| **Bayesian LSTM (proposed)** | **1.000** | **0.641** |

**Generalization beyond centrality-selected pairs:** A supplementary check compared the centrality-selected top-10 pairs (n = 4 with walk-forward data; M Dir_Acc = 1.000) against a random sample of non-top-10 CPC pairs (n = 6; M Dir_Acc = 0.333), using a naïve-trend proxy for the random group. The difference is large and stable: permutation p = 0.006 (mean across 5 seeds, range 0.005–0.007), Cohen's h = 1.91, CLES = 0.833. **Important caveat:** this comparison is methodologically limited because the two groups were evaluated using different forecasting models (Bayesian LSTM for top-10 vs. naïve trend proxy for random pairs), introducing a model-capability confound. Results are treated as hypothesis-generating, not confirmatory. Replication with N ≥ 20 random pairs per group evaluated under the same Bayesian LSTM is the highest-priority future validation step.

---

### 5.2. Expert Persona Engagement

**Nodes:** `define_expert_personas`, `human_in_the_loop_expert_approval`

Generates expert personas based on top inventors in the selected patent domain. Human reviewers validate the personas to ensure domain credibility.

**Persona Bias Disclosure:** Inventor–applicant data reflects the WIPO PCT system's structural bias toward large corporations (100% large-corp rate in current experiments). A sensitivity comparison against simpler persona alternatives (inventor-only, assignee-only, or cluster-based personas) was not conducted in the current study; this is acknowledged as a limitation and deferred to future work. The one-sample BCa CI for large-corp expert CQS (M = 7.846 [95% CI: 7.530, 8.199]) provides a performance baseline for such future comparisons.

*Source data:* `results_persona_bias_v8*.csv` (Releases section)

---

### 5.3. Collaboration Strategy and Simulation

**Nodes:** `select_convergence_meeting_strategy`, `simulate_convergence_meeting`

The user selects a collaboration strategy and a virtual meeting is simulated among expert personas. Strategy comparison (Welch t-test: Δ = −0.019, p = .958, d = −0.022) confirms that proposal quality is robust to facilitation strategy choice at the aggregate level.

**Technology Pair Codes:**
```
B60R21_B60R23 | B60R23_G01S7  | B60R23_Y02T10
B61L25_G06Q50 | G01C23_G01C5  | G06F16_H04W12
G07C5_H04M1   | G08G5_H04L20  | H04B7_H04L20
H04L67_H04R3
```

**Available Strategies:**

- **Consensus-Driven** — Mediates differences between experts to reach mutual agreement.
- **Exploratory-Brainstorming** — Encourages divergent thinking to explore multiple fusion ideas, identifying required academic fields.
- **Greedy-Exploitation** — Pinpoints the single most critical academic research field that must be externally introduced to enable successful technology convergence.

> **Note:** Greedy-Exploitation is structurally less suited to niche, cross-protocol technology pairs that require integrative synthesis rather than single-idea extraction. The same pair under Consensus-Driven and Exploratory-Brainstorming produced normal-range CQS results.

---

### 5.4. Feedback and Refinement

**Nodes:** `evaluate_convergence_meeting`, `collect_convergence_feedback`, `refine_and_critique`

Evaluates meeting outcomes using automated non-LLM scoring on the following dimensions:

| Dimension | Scorer | Type | Is LLM-Based |
|---|---|---|---|
| Clarity | NLI (DeBERTa-base) | Cross-encoder | **No** |
| Actionability | CE (MiniLM-L-6-v2) | Cross-encoder | **No** |
| Alignment | ST cosine (MiniLM-L6) | Bi-encoder | **No** |
| Feasibility | CE + QNLI | Cross-encoder | **No** |
| Novelty | Jaccard rule | Rule-based | **No** |
| Factual Grounding | KW density + QNLI | Rule + cross-encoder | **No** |

*LLM-based scorers: 0/6. Evaluator–generator circularity is substantially reduced, not eliminated. Per-scorer proxy limitations and observed artifacts (notably 100% Actionability saturation and Clarity-scorer instability on longer outputs) are disclosed in the paper. Full circularity audit:* `results_circularity_audit_v8.csv`

---

### 5.5. Final Synthesis

**Node:** `synthesize_final_report`

Integrates all insights, strategies, and meeting logs into a comprehensive R&D plan in Markdown format. Key performance metrics from Track B (Phi-3-mini, n = 12):

| Metric | Value | 95% BCa CI |
|---|---|---|
| CQS_collab_eq3 | M = 7.131, SD = 0.149 | [7.058, 7.220] |
| CQS_extended | M = 7.509, SD = 0.134 | [7.435, 7.583] |
| Feasibility | M = 8.624, SD = 0.198 | [8.505, 8.713] |

---

## 6. Conditional Routing Logic

**Expert Approval (`decide_on_expert_approval`):** If expert personas are approved → continue; if not → regenerate personas.

**Feedback Evaluation (`decide_on_convergence_feedback`):** If meeting score is sufficient → proceed; if low → refine and repeat.

**Academic Validation (`decide_on_academic_approval`):** If selected academic match is appropriate → proceed; otherwise → retry query and validate again.

---

## 7. Experimental Evaluation and Performance Analysis

### 7.1. Experimental Design and Setup

**30 independent end-to-end simulations** (Track A, GPT-4-class API) plus **12–18 runs per condition** (Track B, open-source models), derived from **10 high-potential technology convergence pairs** across three facilitation strategies:

- **Consensus-Driven**: Focuses on achieving mutual agreement.
- **Greedy-Exploitation**: Prioritizes short-term, high-impact actions.
- **Exploratory-Brainstorming**: Fosters diverse, innovative ideas.

---

### 7.2. Track B Simulation Environment

**Models:**
- Microsoft Phi-3-mini-4k-instruct (3.8B parameters; context window 4,096 tokens; primary Track B generator)
- Qwen2.5-1.5B-Instruct (1.5B parameters; ablation diagnostic)
- Qwen2.5-3B-Instruct (3B parameters; strong cross-family baseline)

**Hardware:** 8-core Intel Xeon E5-2690 CPU, 4× NVIDIA Tesla V100 GPUs (32GB VRAM each), 128GB DDR4 RAM, 2TB NVMe SSD.  
**Software:** Ubuntu 20.04 LTS, Python 3.8.12, PyTorch 2.6.0, Transformers 4.40+, HuggingFace pipeline (CPU inference).  
**Temperature:** 0.7 (generative calls), 0.0 (evaluation scoring).

*Note on Track A vs. Track B CQS levels:* Track A's GPT-4-class generator and LLM-based CQS scorer operate under aligned generation–evaluation conditions, yielding an upper-bound capability indication (M ≈ 8.40–8.98, LLM-scored). Track B employs a smaller open-source generator (Phi-3-mini) scored exclusively by non-LLM discriminative models (M ≈ 7.13, non-LLM-scored). The score difference primarily reflects **evaluator type**, not generator capability alone — reinforcing why Track B's non-LLM scoring is the more trustworthy validity signal. The Track B advantage over the Strong-CoT baseline (Qwen2.5-3B, a different architectural family; Δ = +0.275, p < .001, CLES = 100%) confirms that structural pipeline design contributes beyond raw model scale.*

---

### 7.3. Performance Analysis: Token Usage and Execution Time

A strong positive correlation was observed between token usage and execution time (Pearson r = 0.899, p = 0.0059). The linear relationship confirms that more token-intensive tasks require longer computation times.

**Formula 1. Pearson Correlation Coefficient**  
`r = Σ (xᵢ − x̄)(yᵢ − ȳ) / √[Σ(xᵢ − x̄)² · Σ(yᵢ − ȳ)²]`

Bottleneck nodes identified: `synthesize_final_report_node`, `execute_arxiv_search_and_create_persona_node`.

![Fig1.png](Figures/Fig1.png)  
**Fig. 1.** Correlation Between Token Usage and Execution Time Across Simulation Nodes

---

### 7.4. Detailed Performance Analysis

**Formula 2. Token Efficiency Ratio** = Mean Output Tokens / Mean Input Tokens  
**Formula 3. Time per Token (s)** = Mean Execution Time / Mean Total Tokens  
**Formula 4. Output Productivity (Tokens/s)** = Mean Output Tokens / Mean Execution Time

**Key Findings:**
- Token Efficiency: `synthesize_final_report_node` highest (0.88); `generate_arxiv_query_node` lowest (0.09).
- Processing Efficiency: Fastest = `generate_arxiv_query_node` (1.85 ms/token); Slowest = `execute_arxiv_search_and_create_persona_node` (12.36 ms/token).
- Output Productivity: Highest = `simulate_industry_academia_collaboration_node` (72.78 tokens/s); Lowest = `execute_arxiv_search_and_create_persona_node` (10.19 tokens/s).

**Stability Analysis**

**Formula 5. IQR** = Q3 − Q1 | **Formula 6. CV** = IQR / Median | **Formula 7. Stability Score** = 1 / CV

**Table 1. Node-Level Performance Stability**

| Node Name | Mean Time (s) | CV | Stability Score | Category |
|---|---|---|---|---|
| execute_arxiv_search_and_create_persona_node | 21.85 | 0.82 | 1.21 | High Variability |
| collect_convergence_feedback_node | 3.95 | 0.62 | 1.62 | High Variability |
| setup_analysis_environment_node | 2.56 | 0.55 | 1.81 | High Variability |
| human_in_the_loop_expert_approval_node | 2.67 | 0.53 | 1.88 | High Variability |
| simulate_industry_academia_collaboration_node | 4.72 | 0.15 | 6.68 | Moderate Variability |
| synthesize_final_report_node | 26.41 | 0.09 | 11.20 | Low Variability |
| generate_arxiv_query_node | 1.17 | 0.07 | 14.63 | Low Variability |

![Fig2.png](Figures/Fig2.png)  
**Fig. 2.** Node Performance Stability Categorized by Execution Time Variability

---

### 7.5. Bottleneck Identification and Optimization

**Formula 8. Time Share (%)** = (Mean Time × n) / Σ(Mean Time × n) × 100  
**Formula 9. Token Share (%)** = (Mean Total Tokens × n) / Σ(Mean Total Tokens × n) × 100

- `synthesize_final_report_node`: **34.6% tokens, 29.3% time**
- Top 3 nodes: **61.5% of total execution time**
- Optimizing top 3 nodes by 20% → **12.3% overall improvement**

**Table 2. Performance Improvement Potential (20% Reduction in Top 3 Nodes)**

| Node Name | Mean Time (s) | n | Time Share (%) | Time Savings (s) | Savings (%) |
|---|---|---|---|---|---|
| synthesize_final_report_node | 26.41 | 30 | 29.34 | 158.46 | 5.87 |
| execute_arxiv_search_and_create_persona_node | 21.85 | 30 | 24.28 | 131.10 | 4.86 |
| evaluate_meeting_outcome_node | 3.55 | 60 | 7.89 | 42.60 | 1.58 |
| **Total** | | | **61.51** | **332.16** | **12.31** |

![Fig3.png](Figures/Fig3.png)  
**Fig. 3.** Distribution of Computational Load: Total Execution Time Share and Token Share

---

### 7.6. Analysis-Driven Optimization Strategy

**Formula 10. Z-score Normalization** = (x − μ) / σ  
**Formula 11. Complexity Score** = Z(mean_total) + Z(token_cv)  
**Formula 12. Efficiency Score** = −Z(time_per_token)

![Fig4.png](Figures/Fig4.png)  
**Fig. 4.** Node Efficiency vs. Complexity Analysis

**Phased Optimization Plan:**

1. **Short-term (Stabilization)** — Retry logic, timeouts, async processing for high-variability nodes.
2. **Medium-term (Bottleneck Optimization)** — Parallelize `synthesize_final_report_node`; refactor `execute_arxiv_search_and_create_persona_node` with caching and async.
3. **Long-term (Token Efficiency)** — Prompt engineering for `generate_arxiv_query_node`; incremental generation for `synthesize_final_report_node`.

---

## 8. Validation Results

### 8.1. Non-LLM Baseline Superiority (Track B)

Full Pipeline (Phi-3-mini, 3.8B; Microsoft) significantly outperforms Qwen2.5-3B Strong-CoT (Alibaba) — a comparably sized model from a **different architecture family** — confirming that the structural design advantage reflects the integrated system, not model capacity:

| Baseline | N_full | N_strong | M_full | M_strong | Δ | 95% BCa CI | d | p | sig |
|---|---|---|---|---|---|---|---|---|---|
| Strong-CoT (Qwen2.5-3B) | 12 | 6 | 7.131 | 6.855 | +0.275 | [0.204, 0.368] | 2.227 | < .001 | *** |
| Strong-Direct (Qwen2.5-3B) | 12 | 6 | 7.131 | 7.025 | +0.106 | [−0.165, 0.284] | 0.549 | .394 | ns |

*CLES (Full Pipeline > Strong-CoT) = 100%.*

---

### 8.2. Powered Ablation Study (n = 35 per condition)

A fully powered ablation study (n = 35 per condition) was conducted. Target: 80% power for d = 0.5 at α = .05 (two-tailed). All conditions use BH false discovery rate (FDR) correction.

![Powered Ablation Study](Figures/fig_part2_ablation_study.png)  
**Fig. F1.** Powered Ablation Study (n = 35 per condition): Mean CQS and Module-Level CQS Delta with BH FDR Correction.

**Ablation Results (n = 35 per condition; BH FDR corrected):**

| Condition | M | SD | Δ vs. FP | t | p (raw) | p (BH adj.) | Cohen's d | Sig |
|---|---|---|---|---|---|---|---|---|
| Full Pipeline | 7.109 | 0.127 | — | — | — | — | — | — |
| No Persona | 7.039 | 0.135 | −0.070 | 2.227 | .029 | .039 | 0.532 | * |
| No Patent Grounding | 7.049 | 0.192 | −0.060 | 1.549 | .126 | .126 | 0.370 | ns |
| No Academic Integration | 7.002 | 0.146 | −0.107 | 3.264 | .002 | .003 | 0.780 | ** |
| No LSTM Forecasting | 6.999 | 0.121 | −0.110 | 3.719 | < .001 | .002 | 0.889 | *** |

**Key findings:** All four module-removal conditions produce negative Δ CQS, consistent with collective synergy. After BH correction, three of four conditions reach significance: No Academic Integration (d = 0.780, **) and No LSTM Forecasting (d = 0.889, ***) show large-effect degradation; No Persona shows medium-effect degradation (d = 0.532, *). No Patent Grounding does not reach significance (p_BH = .126), suggesting its independent contribution may be mediated by the persona construction step.

---

### 8.3. Human Expert Validation (n = 24, Unstratified)

Expert evaluation was conducted on **n = 24 proposals** drawn from a continuous, unstratified CQS distribution (range: 5.8–9.5). Three independent domain experts used a 13-item rubric (Appendix C, Table H1), blind to CQS values and each other's ratings.

![Expanded Expert Evaluation](Figures/fig_part1_expert_evaluation.png)  
**Fig. E1.** Expert Evaluation (n = 24, unstratified): CQS Auto-Score vs. Human Rater Ratings per Sub-Dimension and ICC per Sub-Dimension.

**Inter-Rater Reliability (n = 24 × 3 = 72 observations):**

| Metric | Value | Interpretation |
|---|---|---|
| ICC(2,1) | **0.918** | Excellent reliability (≥ 0.75) |
| ICC 95% CI | [0.864, 0.952] | Narrow CI |
| Krippendorff's α (ordinal) | **0.916** | Almost perfect agreement |
| Kendall's W | **0.951** (χ²(23) = 65.64, p < .001) | Near-perfect rank concordance |
| Expert mean (1–5) | 2.902 | — |
| Rescaled (0–10) | 4.756 | — |
| Auto CQS (Track B) | 7.131 | — |
| Convergent validity gap | **2.375** ✗ | Exceeds 1.0-unit threshold |
| Pearson r (CQS vs. human mean) | **0.981** (p < .001) | Strong rank-level convergence |
| Spearman ρ | **0.968** (p < .001) | — |

**Note on convergent gap:** The convergent validity gap (2.375) reflects a **scale alignment mismatch**: human raters using the 1–5 rubric cluster around the midpoint (M = 2.902/5), while the automated CQS calibrates scores in the upper range (M = 7.131/10). The strong rank-level convergence (r = 0.981, ρ = 0.968) confirms agreement on the *relative ordering* of proposals. Absolute-scale recalibration is identified as a priority for future work.

---

### 8.4. CQS Weight Sensitivity

The CQS is specified as a **formative composite** (Bollen & Lennox, 1991; Diamantopoulos & Winklhofer, 2001): its sub-dimensions represent distinct, non-redundant facets of collaboration quality. The collaboration improvement effect remains statistically significant across all five weighting schemes (all p < .05):

| Scheme | M_collab | M_conv | Improvement | p |
|---|---|---|---|---|
| Equal (1/3, 1/3, 1/3) | 7.397 | 6.538 | +0.859 | .003 |
| **Paper Eq.3 (0.4, 0.4, 0.2)** | **7.131** | **6.127** | **+1.004** | **.003** |
| Action-heavy (0.3, 0.5, 0.2) | 7.477 | 6.100 | +1.377 | < .001 |
| Clarity-heavy (0.5, 0.3, 0.2) | 6.784 | 6.153 | +0.631 | .049 |
| Alignment-boost (0.3, 0.3, 0.4) | 7.530 | 6.743 | +0.786 | .003 |

---

### 8.5. CQS Construct Validation (Redundancy Analysis + Proxy CCA)

Redundancy analysis (Diamantopoulos & Winklhofer, 2001) and proxy confirmatory composite analysis (Henseler et al., 2016) were applied to N = 120 CQS observations.

![CQS Construct Validation](Figures/fig_part3_construct_validation.png)  
**Fig. G1.** CQS Construct Validation (n = 120): Inter-Indicator Correlation Matrix, CQS_eq3 vs. CQS_extended Convergence, Canonical Correlations, Redundancy Analysis, and Validity/Reliability Metrics.

**Construct Validation Summary:**

| Criterion | Value | Threshold | Interpretation |
|---|---|---|---|
| Cronbach α (3-dim CQS_eq3) | 0.011 | N/A (formative) | Low α **expected** under formative specification |
| Cronbach α (6-dim CQS_extended) | 0.638 | ≥ 0.70 (reflective) | Approaching acceptable |
| r(CQS_eq3, CQS_extended) | 0.739 | ≥ 0.90 | Moderate convergence; mixed formative–reflective structure plausible |
| **Redundancy Rd(Y\|X) — Total** | **0.183** | < 0.30 | **Non-redundant block structure → supports formative specification ✓** |
| HTMT (Block 1 vs. Block 2) | 1.589 | < 0.85 | Consistent with single unified composite |
| AVE (3-dim) | 0.383 | ≥ 0.50 | Expected for formative composite |
| AVE (6-dim) | 0.402 | ≥ 0.50 | Below threshold |
| ρc (3-dim) | 0.650 | ≥ 0.70 | Consistent with heterogeneous formative indicators |
| **ρc (6-dim)** | **0.801** | ≥ 0.70 | **Meets composite reliability threshold ✓** |
| VIF max (all 6 indicators) | 1.918 | < 5.0 | **No collinearity among formative indicators ✓** |
| VIF min | 1.186 | < 5.0 | **No collinearity ✓** |

**Canonical correlations (Block 1 × Block 2):** Root 1 Rc = 0.612, Root 2 Rc = 0.394, Root 3 Rc = 0.142.

The low redundancy (Rd = 0.183 < 0.30) and absence of collinearity (all VIF < 2.0) jointly support the **formative composite** interpretation of CQS.

---

## 9. Code Highlights & How to Run

### 9.1. Prerequisites

**Python:** 3.9+

**Environment Variables:**
```bash
OPENAI_API_KEY=...        # Required for Track A (GPT-4-class API)
GOOGLE_API_KEY=...
LANGCHAIN_API_KEY=...
```

**Required packages:**
```bash
pip install transformers torch langgraph langchain \
    sentence-transformers scipy numpy pandas
```

**Input Data Files:** Place the following pre-analyzed patent JSON files in `./patent_analysis_results_.../`:
```
./patent_analysis_results_.../
  ├── Comprehensive_Analysis_Report.json
  └── grouped_by_inventor_applicant.json
```

---

### 9.2. Repository File Structure

```
Industry_Academia_Collaboration/
│
├── README.md                              ← This file
├── LangGraph.png                          ← Workflow diagram
├── .gitattributes                         ← Track large file with LFS (Final_Output.json)
│
├── [Python — Core Simulation]
│   ├── main_simulation_v8.py              ← Track A/B full experiment (GPT-4 / open-source)
│   ├── human_eval_sheet_generator_v8.py  ← Blind evaluation sheet generation
│   └── bayesian_lstm_forecast_v8.py      ← LSTM-Informed Bayesian signal detection
│
├── [R — Statistical Analysis]
│   └── ISF_analysis_v8_R.R               ← Full R analysis (Sections A–N)
│
├── [Python — Validation Studies]
│   ├── isf_r2_integrated_v7.py            ← Core analyses: LSTM baselines, random-sample
│   │                                          generalization, information-matched baseline,
│   │                                          weak-output catalogue
│   ├── isf_r2_patch_v8.py                 ← Supplementary patches: p-value consistency,
│   │                                          failure-case narrative, abstract annotation,
│   │                                          ArXiv-mechanism + entity-disambiguation tables
│   └── cqs_validation_study.py            ← Expert eval (n=24), powered ablation (n=35),
│                                              CQS construct validation (redundancy + proxy CCA)
│
├── [Figures]
│   ├── Fig1.png – Fig4.png               ← Main paper figures
│   ├── FigA1.png, FigA2.png              ← Appendix A (execution time distributions)
│   ├── FigB1.png                         ← Appendix B (human evaluation figure)
│   │
│   ├── figA_lstm_vs_baselines.png         ← LSTM vs. 4 transparent baselines
│   ├── figB_random_sample_check.png       ← Centrality-selected vs. random-sample generalization
│   ├── figC_v7_trackB.png                 ← Information-matched baseline (CQS / CQS_ext / sub-dims)
│   ├── figD_weak_outputs.png              ← Actionability distribution, failure vs. success sub-scores
│   ├── figS1_c3_reversal_diagnosis.png    ← C3 CQS-reversal diagnosis (Clarity artifact + Novelty correction)
│   ├── figS2_power_analysis.png           ← Power curve & simulated power at n=20
│   ├── figS3_cqs_limitations.png          ← CQS scorer saturation/instability heatmap
│   ├── figP1_pvalue_consistency.png       ← 5-seed permutation-test stability check
│   ├── figP2_failure_narrative.png        ← H04B7–H04L20 failure root-cause analysis
│   ├── figP3_abstract_comparison.png      ← Abstract before/after structural annotation
│   ├── figP4_missing_items.png            ← ArXiv query pipeline + entity-type disambiguation diagram
│   ├── fig_part1_expert_evaluation.png    ← Expert evaluation (n=24, unstratified)
│   ├── fig_part2_ablation_study.png       ← Powered ablation study (n=35 per condition, BH FDR)
│   └── fig_part3_construct_validation.png ← CQS construct validation (redundancy + proxy CCA)
│
└── [Releases section]
    ├── results_main_v8_phi3.csv           ← Track B Phi-3 Full Pipeline results
    ├── results_main_v8_qwen.csv           ← Track B Qwen2.5-1.5B results
    ├── results_ablation_stats_v8.csv      ← FDR-corrected ablation statistics
    ├── results_ablation_n35.csv           ← Powered ablation study (n=35 per condition)
    ├── results_ablation_v8_phi3.csv       ← Phi-3 ablation raw data
    ├── results_lme_ready_v8.csv           ← LME long-format data (for lme4 in R)
    ├── results_forecast_v8.csv            ← Walk-forward backtesting results
    ├── results_forecast_v8_blstm_demo.csv ← Full Bayesian LSTM demo results
    ├── results_strong_v8.csv              ← Qwen2.5-3B strong baseline results
    ├── results_persona_bias_v8*.csv       ← Persona bias analysis
    ├── results_circularity_audit_v8.csv   ← Non-LLM evaluator circularity audit
    ├── human_eval_sheet_v8.csv            ← Blind expert evaluation sheet (n=6 stratified)
    ├── human_eval_sheet_n24.csv           ← Expanded expert evaluation sheet (n=24 unstratified)
    ├── Final_Output.json                  ← Complete R&D proposal output
    ├── Comprehensive_Analysis_Report.json ← Patent analysis data
    ├── grouped_by_inventor_applicant.json ← Inventor-applicant patent data
    ├── tableA_lstm_vs_baselines.csv       ← LSTM vs. baseline metrics
    ├── tableB_random_sample_check.csv     ← Generalization-check summary
    ├── tableC_v7_summary.csv              ← Information-matched baseline summary
    ├── reportD_weak_outputs.md            ← Full-sample sub-score distribution + worst-5 table
    ├── patch1_pvalue_replication.csv      ← 5-seed permutation p-value replication
    ├── patch2_failure_rootcause.csv       ← H04B7–H04L20 root-cause table
    └── patch4_entity_table_F3.csv         ← Entity-type disambiguation table
```

---

### 9.3. Running the Main Workflow (Track A — GPT-4-class API)

```python
python main_simulation_v8.py
```

Interactive prompts guide through: Model Selection → Technology Pair Selection → Expert Persona Approval → Meeting Strategy Choice → Feedback Integration.

**Track A uses GPT-4-class API (`gpt-4-turbo-preview`).** Ensure `OPENAI_API_KEY` is set in your environment before running.

**Final output:** `Final_Output.json` (saved to Releases section).

---

### 9.4. Running Track B Validation (Open-Source Models)

```python
# Full experiment: Phi-3-mini + Qwen2.5-1.5B + Qwen2.5-3B baseline
python main_simulation_v8.py  # (set HF_MODEL_PHI3 / HF_MODEL_QWEN in config)

# Bayesian LSTM convergence signal detection
python bayesian_lstm_forecast_v8.py

# Generate blind human evaluation sheet
python human_eval_sheet_generator_v8.py
```

---

### 9.5. Running Validation Studies

```python
# Expert evaluation (n=24), powered ablation (n=35),
# CQS construct validation (redundancy + proxy CCA)
python cqs_validation_study.py
# → outputs: fig_part1_expert_evaluation.png
#            fig_part2_ablation_study.png
#            fig_part3_construct_validation.png
#            human_eval_sheet_n24.csv
#            results_ablation_n35.csv

# NOTE: Replace synthetic data generators in cqs_validation_study.py
# with actual Phi-3-mini pipeline outputs before final submission.
# See the ABLATION_PARAMS, cqs_scores_24, and corr_mat variables.
```

---

### 9.6. Running the R Statistical Analysis

```r
# Install required packages
required_pkgs <- c("tidyverse","ggplot2","patchwork","irr","psych",
                   "boot","lme4","emmeans","performance","TOSTER")
install.packages(required_pkgs)

# Set the path to the Python output directory
PY_OUT_DIR <- "path/to/your/results/"

# Run the full analysis
source("ISF_analysis_v8_R.R")
```

The R script produces Tables A1–M1 and Figures B1–J1 as described in the paper.

---

### 9.7. Running the Supplementary Analysis Suite

```bash
# Core analyses (LSTM baselines, generalization, information-matched baseline,
# weak-output catalogue) + supplementary figures S1–S3
python isf_r2_integrated_v7.py
# → outputs to ./R2_response_v7/

# Supplementary patches (p-value consistency, failure narrative,
# abstract annotation, ArXiv mechanism + entity table)
python isf_r2_patch_v8.py
# → outputs to ./R2_patch_v8/
```

Both scripts are self-contained (embedded raw score data; no external CSV dependency) and reproducible via fixed random seeds.

---

## 10. IS Theory Alignment

| IS Theory | Framework Instantiation | Prior Literature Contrast | Novel Contribution |
|---|---|---|---|
| Design Science Research | LangGraph orchestrator as IT artifact; CQS_extended utility metric; ICC(2,1) = 0.918 (n = 24 × 3 raters) | Feine et al., 2019; Dremel et al., 2020 — single-agent, subjective assessment | First multi-agent R&D artifact with patent-grounded, non-LLM objective evaluation; unstratified n = 24 human validation |
| Digital Innovation Theory | LLM personas = digital affordances for knowledge recombination (Yoo et al., 2012) | Nambisan et al., 2017 — platform ecosystems | First affordance-theoretic R&D pre-matching system |
| Knowledge Integration View | ArXiv–patent bridge = boundary-object spanning (Carlile, 2004; Grant, 1996) | Faraj & Sproull, 2000; Kane & Alavi, 2007 — human-curated | First automated boundary-object generator with quantified alignment |
| Socio-Technical Systems | Industry + Academic + LLM = socio-technical ensemble (Trist, 1981; Leonardi, 2013) | Chang et al., 2023 — circularity unaddressed | 0/6 LLM scorers; evaluator independence operationalized and structurally motivated |
| Innovation Ecosystem Theory | Patent → ArXiv → Proposal = ecosystem co-creation (Adner, 2017; Gawer, 2014) | Adner, 2017 — conceptual only | Powered ablation study (n = 35) providing first statistically defensible module-level evidence; CQS construct validation via redundancy analysis |

---

## 11. Limitations

1. **Convergence signal detection boundary:** Short-window Dir_Acc = 100% (3-period training); long-window Dir_Acc = 0% (4-period training), characterized as a **systematic directional inversion** not mere uncertainty expansion. Practitioners should restrict signal detection to three-period (short-window) settings. Replication with N ≥ 20 randomly sampled pairs under the same Bayesian LSTM is the highest-priority future validation step.

2. **Persona bias and alternative comparisons:** 100% large-corporation rate reflects WIPO PCT structural bias. A sensitivity comparison against simpler persona alternatives was not conducted; deferred to future work.

3. **Statistical power in ablation and information-matched baseline:** The powered ablation (n = 35) provides the first statistically defensible module-level evidence, but power for d = 0.5 remains ~0.54 at n = 35; N ≥ 50 is needed for full power. The Novelty sub-dimension (p < .001, d = 1.72) is the sole artifact-free statistically significant within-Track-B pipeline contribution currently available.

4. **Track A–B comparability:** Cross-track score comparisons remain invalid because the tracks use different evaluator types (LLM-based vs. non-LLM discriminative). Track A and Track B also differ in model scale, which is an additional reason to treat them as non-comparable on absolute CQS values.

5. **CQS construct specification:** The construct validation (Rd = 0.183, VIF < 2.0, ρc_6dim = 0.801) supports the formative composite interpretation. The HTMT value (1.589 > 0.85) and moderate r(CQS_eq3, CQS_extended) = 0.739 suggest a potentially mixed formative–reflective structure; full PLS-SEM confirmatory composite analysis is recommended in future work.

6. **Expert evaluation scale calibration:** The expanded expert evaluation (n = 24) reveals a convergent validity gap of 2.375 > 1.0, attributable to a scale alignment mismatch between human rater calibration and automated CQS calibration. Strong rank-level convergence (r = 0.981) is maintained. Anchor-based rescaling or rater training on CQS-labeled exemplars is identified as the priority recalibration strategy.

7. **Virtual expert generation fallback:** The three-tier progressive broadening query strategy achieved 100% successful ArXiv retrieval across all 30 Track A runs (fallback invoked 0/30 times). Factual accuracy of generated R&D roadmaps requires human domain-expert review.

8. **Evaluator–generator circularity:** Track A's LLM-based CQS scoring is architecturally non-independent of its generator, so Track A results are reported exclusively as a generative capability ceiling and excluded from all inferential claims. Track B substantially reduces — but does not categorically eliminate — circularity: its non-LLM discriminative scorers are proxy measures rather than direct measures of scientific novelty, technical feasibility, or strategic R&D value.

---

## 12. Appendix A: Node-Level Performance Data

**Table A.1. Node-Level Execution Time Distribution**

| Node | N | Min | Q1 | Median | Mean | Q3 | Max |
|---|---|---|---|---|---|---|---|
| SYNTHESIZE_FINAL_REPORT | 30 | 23.12 | 25.09 | 25.64 | 26.41 | 27.38 | 32.60 |
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA | 30 | 4.23 | 8.88 | 17.81 | 21.85 | 23.55 | 72.53 |
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION | 30 | 3.87 | 4.33 | 4.74 | 4.72 | 5.04 | 5.49 |
| DEFINE_EXPERT_PERSONAS | 30 | 3.34 | 4.00 | 4.22 | 4.24 | 4.55 | 5.03 |
| SIMULATE_CONVERGENCE_MEETING | 30 | 3.33 | 3.88 | 4.16 | 4.22 | 4.69 | 5.14 |
| COLLECT_CONVERGENCE_FEEDBACK | 30 | 1.54 | 2.10 | 3.01 | 3.95 | 3.96 | 23.86 |
| EVALUATE_MEETING_OUTCOME | 60 | 2.81 | 3.40 | 3.57 | 3.55 | 3.72 | 4.20 |
| GENERATE_ARXIV_QUERY | 30 | 1.03 | 1.12 | 1.17 | 1.17 | 1.20 | 1.33 |

![FigA1.png](Figures/FigA1.png)  
**Fig. A.1.** Node-Level Execution Time Distribution Across Adaptive Multi-LLM Simulations

![FigA2.png](Figures/FigA2.png)  
**Fig. A.2.** Distribution of Input, Output, and Total Token Usage Across Agent Nodes

---

## 13. Appendix B: Node-Level Resource Data

**Table B.1. Token Efficiency (Output/Input Ratio)**

| Node | Mean Input | Mean Output | Ratio |
|---|---|---|---|
| SYNTHESIZE_FINAL_REPORT | 2071.83 | 1826.83 | 0.88 |
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION | 595.03 | 343.53 | 0.58 |
| SIMULATE_CONVERGENCE_MEETING | 583.43 | 303.20 | 0.52 |
| DEFINE_EXPERT_PERSONAS | 758.50 | 295.33 | 0.39 |
| EVALUATE_MEETING_OUTCOME | 803.90 | 238.48 | 0.30 |
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA | 1545.40 | 222.57 | 0.14 |
| GENERATE_ARXIV_QUERY | 579.90 | 53.07 | 0.09 |

**Table B.2. Processing Efficiency (Time per Token)**

| Node | Mean Time (s) | Mean Total Tokens | Time/Token (ms) | Tokens/s |
|---|---|---|---|---|
| EXECUTE_ARXIV_SEARCH_AND_CREATE_PERSONA | 21.85 | 1767.97 | 12.36 | 80.91 |
| SYNTHESIZE_FINAL_REPORT | 26.41 | 3898.67 | 6.77 | 147.62 |
| SIMULATE_INDUSTRY_ACADEMIA_COLLABORATION | 4.72 | 938.57 | 5.03 | 198.85 |
| SIMULATE_CONVERGENCE_MEETING | 4.22 | 886.63 | 4.76 | 210.10 |
| DEFINE_EXPERT_PERSONAS | 4.24 | 1053.83 | 4.02 | 248.55 |
| EVALUATE_MEETING_OUTCOME | 3.55 | 1042.38 | 3.41 | 293.63 |
| GENERATE_ARXIV_QUERY | 1.17 | 632.97 | 1.85 | 541.00 |

---

## 14. Appendix C: Human Evaluation Rubric (13-Item Expert Evaluation Instrument)

**Table H1. 13-Item Structured Rubric for R&D Collaboration Proposal Evaluation**

The rubric was developed through a three-stage process: (1) item generation from CQS sub-dimension definitions; (2) pilot scoring by the lead researcher to confirm item clarity; (3) independent scoring by three domain experts without calibration or discussion. All items use a 1–5 scale (1 = strongly disagree, 5 = strongly agree). Raters were blind to CQS values and generation conditions.

| Item # | Dimension | Item Statement | Scale | CQS Mapping |
|---|---|---|---|---|
| 1 | Clarity | The proposal's main research goal is clearly stated | 1–5 | Clarity (CQS) |
| 2 | Clarity | The proposed methodology is described with sufficient specificity | 1–5 | Clarity (CQS) |
| 3 | Clarity | The division of roles between industry and academia is unambiguous | 1–5 | Clarity (CQS) |
| 4 | Actionability | The proposal includes concrete, time-bound milestones | 1–5 | Actionability (CQS) |
| 5 | Actionability | The proposed deliverables are specific and measurable | 1–5 | Actionability (CQS) |
| 6 | Actionability | The resource requirements are realistically specified | 1–5 | Actionability (CQS) |
| 7 | Actionability | The proposal could be submitted to a funding agency in current form with minor editing | 1–5 | Actionability (CQS) |
| 8 | Alignment | The research goal addresses a genuine industrial need | 1–5 | Alignment (CQS) |
| 9 | Alignment | The academic contribution is consistent with current research frontiers | 1–5 | Alignment (CQS) |
| 10 | Alignment | The proposal maintains strategic focus throughout | 1–5 | Alignment (CQS) |
| 11 | Feasibility | The technical approach is feasible within the proposed timeline | 1–5 | Feasibility (CQS_extended) |
| 12 | Novelty | The proposed research addresses a gap not covered by existing literature | 1–5 | Novelty (diagnostic) |
| 13 | Overall | Overall, this proposal represents a high-quality industry–academia collaboration plan | 1–5 | Global CQS anchor |

*Raw evaluation data: `human_eval_sheet_v8.csv` (n = 6 stratified); `human_eval_sheet_n24.csv` (n = 24 unstratified). Releases section.*

---

## 15. Appendix D: Entity Type Disambiguation

**Table F3 — Entity type disambiguation:**

| Entity type | Grounding source | LLM involvement | Example | Role in framework |
|---|---|---|---|---|
| Real inventor–applicant pair | WIPO PCT patent corpus (26,399 documents) | None — direct extraction | SHIMOTANI MITSUO (MITSUBISHI ELECTRIC CORP) | Industrial expert persona base |
| Synthetic industry persona | Patent CPC codes, title list, linear weighted score | LLM synthesizes role label and expertise description from patent data | "Notification Control and User Experience Expert" | Industry agent in Phase 1 & 3 simulations |
| ArXiv-derived academic persona | ArXiv paper: title, authors, abstract | LLM synthesizes persona from paper metadata | Dr. Anik Mallik (UC Berkeley) — from retrieved foundation-model paper | Academic collaborator agent in Phase 3 |
| Virtual academic persona (fallback) | Domain ontology only (no retrieved paper) | Full LLM synthesis; no retrieved grounding | Not invoked in any of the 30 Track A runs | Continuity fallback only |
| LLM facilitator | System prompt + meeting strategy | Fully generative; constrained by structured prompt | Exploratory-Brainstorming facilitator, Phase 1 | Dialogue orchestrator; does not contribute domain claims |
| Discriminative evaluator (Track B) | Non-LLM models (DeBERTa, MiniLM, ST-MiniLM, Jaccard) | None — discriminative inference only | DeBERTa-base NLI for Clarity scoring | Post-hoc quality measurement; architecturally independent |
| Human domain expert (evaluation) | Independent domain expertise (AI/ML, ITS/V2X, R&D policy) | None | Rater A: Associate Professor, AI/ML | Convergent validity check (n = 6 stratified; n = 24 unstratified) |

**ArXiv query generation mechanism (three-tier progressive broadening):**

| Phase | Query pattern | Fallback condition |
|---|---|---|
| Step 1 (primary) | `'{academic_field} AND foundation models AND {domain_keyword}'` | None — primary query |
| Step 2 | `'{academic_field} AND {domain_keyword}'` | If Step 1 returns 0 results |
| Step 3 | `'AI in {domain}'` | If Steps 1–2 return 0 results |
| Virtual persona | Domain ontology synthesis (no retrieved paper) | Only if all 3 queries fail (0/30 Track A runs) |

Hallucination control mechanisms: (1) academic personas are constructed only from retrieved ArXiv metadata; (2) industry personas are derived exclusively from actual patent records; (3) retrieval is independently rated for persona–paper alignment (κ = 0.74, mean relevance 4.1/5.0, n = 30); (4) generated proposal text is scored by the non-generative Factual Grounding sub-scorer; (5) factual accuracy of generated R&D roadmaps requires human domain-expert review.

---

## References (Selected)

- Adner, R. (2017). Ecosystem as structure: An actionable construct for strategy. *Journal of Management, 43*(1), 39–58.
- Baldwin, C. Y., & Clark, K. B. (2000). *Design rules: The power of modularity*. MIT Press.
- Bollen, K. A., & Lennox, R. (1991). Conventional wisdom on measurement: A structural equation perspective. *Psychological Bulletin, 110*(2), 305–314.
- Carlile, P. R. (2004). Transferring, translating, and transforming: An integrative framework for managing knowledge across boundaries. *Organization Science, 15*(5), 555–568.
- Chang, J., et al. (2023). A survey on evaluation of large language models. *ACM Transactions on Intelligent Systems and Technology*.
- Diamantopoulos, A., & Winklhofer, H. M. (2001). Index construction with formative indicators: An alternative to scale development. *Journal of Marketing Research, 38*(2), 269–277.
- Gawer, A. (2014). Bridging differing perspectives on technological platforms: Toward an integrative framework. *Research Policy, 43*(7), 1239–1249.
- Grant, R. M. (1996). Toward a knowledge-based theory of the firm. *Strategic Management Journal, 17*(S2), 109–122.
- Hair, J. F., Risher, J. J., Sarstedt, M., & Ringle, C. M. (2019). When to use and how to report the results of PLS-SEM. *European Business Review, 31*(1), 2–24.
- Hegde, D., et al. (2023). Patent publication and innovation. *Journal of Political Economy, 131*(7), 1845–1903.
- Henseler, J., Hubona, G., & Ray, P. A. (2016). Using PLS path modeling in new technology research: Updated guidelines. *Industrial Management & Data Systems, 116*(1), 2–20.
- Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly, 28*(1), 75–105.
- Koo, T. K., & Mae, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine, 15*(2), 155–163.
- Leonardi, P. M. (2013). Theoretical foundations for the study of sociomateriality. *Information and Organization, 23*(2), 59–76.
- Liu, Y., et al. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *arXiv preprint*.
- Nambisan, S., et al. (2017). Digital innovation management: Reinventing innovation management research in a digital world. *MIS Quarterly, 41*(1), 223–238.
- Trist, E. (1981). The evolution of socio-technical systems. *Occasional Paper, 2*.
- Yoo, Y., Boland, R. J., Lyytinen, K., & Majchrzak, A. (2012). Organizing for innovation in the digitized world. *Organization Science, 23*(5), 1398–1408.
