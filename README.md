# LangGraph Workflow for Adaptive R&D Decision Processes

## 1. Overview

This repository presents a **LangGraph-based multi-agent framework** engineered for **dynamic and adaptive decision-making** in **industry–academia R&D collaboration**. By integrating **Large Language Models (LLMs)** with **structured, data-driven, agentic processes**, this framework simulates and optimizes real-world R&D planning through a Two-Track experimental architecture.

> **Contribution:** A step toward treating early-stage R&D innovation as a simulatable, data-grounded process — advancing, rather than definitively resolving, systematic R&D foresight.

The workflow features **expert-driven approvals**, **iterative feedback loops**, **conditional routing**, and a **non-LLM evaluation architecture** that substantially reduces evaluative circularity — addressing a critical gap in prior LLM-based innovation studies (Chang et al., 2023; Liu et al., 2023).

> **Note on the term "forecasting":** The term appears in the paper title in reference to the framework's broader purpose of R&D trajectory simulation. Within the methodology, the Bayesian LSTM component is more precisely characterized as *convergence signal detection* — identifying short-window directional growth signals rather than producing quantitative point forecasts. This distinction is maintained consistently throughout the paper and this repository.

> **R2 status (2026-06):** This repository reflects the Round-2 (R2) major-revision response to *Information Systems Frontiers* (manuscript #ISFI-D-25-01489R1, Reviewer #4). All R2 analyses, corrected statistics, and the resulting manuscript-level changes are summarized in **§7a–§7d** and the response-letter assets listed in §9.2. The Track A generator model identification has been corrected throughout this README from "GPT-4-class" to **Qwen2.5-3B-Instruct** following a provenance audit (§7d).

---

## Visual Workflow Overview

![Workflow Visualization](LangGraph.png)

---

## 2. Key Features

**Data-Driven Persona Generation** — Automatically identifies and creates expert personas grounded in real-world patent inventor data (WIPO PCT), combined with ArXiv-based academic expert discovery.

**Convergence Signal Detection** — A Bayesian LSTM model (LSTM-Informed Bayesian Log-Linear Regression) identifies high-potential CPC technology convergence pairs using walk-forward validation, benchmarked against four transparent baselines (Naïve Random Walk, Naïve Trend, Linear Regression, ARIMA(1,1,0); see §7a). Reframed from "forecasting" to *convergence signal detection* to accurately reflect its primary function: short-window directional signal identification rather than quantitative point forecasting.

**Non-LLM Evaluation Architecture** — All six CQS scorers are discriminative, non-generative models (NLI DeBERTa, cross-encoder MiniLM, ST cosine similarity, QNLI, Jaccard rule, keyword density). This substantially reduces — though does not fully eliminate — the "LLM-as-judge" circularity problem (Chang et al., 2023; Liu et al., 2023). Scorer-specific limitations (saturation, instability) are disclosed in §7c and Table E1b.

**Human-In-The-Loop (HITL) Oversight** — Key decisions are validated through expert review at multiple nodes, ensuring strategic alignment.

**Iterative Refinement** — Simulated meetings are evaluated, critiqued, and refined until a high-quality R&D proposal emerges.

**Multi-LLM Robustness** — Supports GPT-4, Phi-3-mini, Qwen2.5 (1.5B/3B), and Gemini, enabling comparative analysis across model scales. The Track A generator used in all 30 archived primary runs is **Qwen2.5-3B-Instruct** (corrected from an earlier "GPT-4-class" description; see §7d).

**Detailed Logging & Monitoring** — Captures execution time, token usage, and decisions per node for performance optimization.

---

## 3. Two-Track Experimental Architecture

Empirical contributions are organized as a complementary Two-Track design, where each track answers a different validity question.

| | Track A | Track B |
|---|---|---|
| **Purpose** | Framework capability demonstration | Reproducibility + independent validation |
| **Sections** | Sections 4–5 of paper | Appendix E of paper |
| **Models** | Qwen2.5-3B-Instruct (corrected; see §7d) | Phi-3-mini-4k (generator); Qwen2.5-3B (baseline) |
| **Run count** | 30 | 12–18 per condition |
| **CQS scorer** | LLM-based (original study) | Non-LLM: 6 discriminative models |
| **CQS range** | M = 8.40 → 8.98 (paired improvement) | M = 7.131 (Full Pipeline) |
| **Key output** | Significant improvement (p < .001, d = 2.62) | Baseline superiority (p < .001, d = 2.23) |
| **Circularity** | Generator and LLM-based evaluator share a model family — circularity not addressed | **Substantially reduced: 0/6 LLM scorers** |
| **Human validation** | Not conducted | ICC(2,1) = 0.898; Kendall W = 1.000 |

*Cross-track score comparisons are not appropriate given model capability differences. Both tracks support the main claim through independent evidentiary pathways. Track A characterizes generative capability under aligned generation–evaluation conditions; Track B is the primary evidence base for validity, since its generator (Phi-3-mini) is architecturally independent of its evaluators (six non-LLM discriminative scorers).*

---

## 4. Contribution Highlights

Workflow design, visualization development, analysis coordination, R2 revision-response analysis, and results documentation were all conducted by **Yong-Jae Lee**.

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

*Primary metric is directional accuracy (Dir_Acc). All Theil's U < 1.0 across all 12 windows (beats naïve baseline on all pairs and windows). The long-window Dir_Acc = 0% result is treated as a **systematic directional inversion**, not a generic "uncertainty boundary": if uncertainty expansion alone were responsible, accuracy would converge toward ~50% (chance), not 0%. R2 analysis (§7a) attributes this to overfitting against an anomalous 2015–2019 growth surge induced by the 18-month patent-publication lag. The 80% PI coverage remains 100% even on long windows, confirming that uncertainty calibration is preserved despite the directional bias. All practitioner guidance is now restricted to the three-period (short-window) setting. Separate full Bayesian LSTM demonstration with MC-dropout uncertainty quantification (H04L67–H04R3 demo pair, Table D3): Dir_Acc = 100%, SMAPE = 20.7%, Theil's U = 0.609.*

**R2 addition — LSTM vs. transparent baselines (§7a):** The short-window Bayesian LSTM (Dir_Acc = 1.000, Theil's U = 0.641) was benchmarked against four transparent alternatives evaluated on the same six pairs:

| Method | Dir_Acc | Theil's U |
|---|---|---|
| Naïve Random Walk | 0.000 | 0.901 |
| Naïve Trend | 0.833 | 0.769 |
| Linear Regression | 0.833 | 0.839 |
| ARIMA(1,1,0) | 0.333 | 1.008 (fails the random-walk test) |
| **Bayesian LSTM (proposed)** | **1.000** | **0.641** |

**R2 addition — generalization beyond centrality-selected pairs (§7a):** A supplementary check compared the centrality-selected top-10 pairs (n = 4 with walk-forward data; M Dir_Acc = 1.000) against a random sample of non-top-10 CPC pairs (n = 6; M Dir_Acc = 0.333), using a naïve-trend proxy for the random group. The difference is large and stable: permutation p = 0.006 (mean across 5 seeds, range 0.005–0.007), Cohen's h = 1.91, CLES = 0.833, with non-overlapping BCa 95% CIs (Top-10 [1.00, 1.00] vs. Random [0.00, 0.67]). *An earlier internal draft of this check reported p = 0.076; that figure was a transcription artifact from an interim run and has been superseded by the validated p = 0.006 result reported here and in the manuscript.* N remains small (4, 6); replication with N ≥ 20 random pairs per group is the top-priority future validation step.

---

### 5.2. Expert Persona Engagement

**Nodes:** `define_expert_personas`, `human_in_the_loop_expert_approval`

Generates expert personas based on top inventors in the selected patent domain. Human reviewers validate the personas to ensure domain credibility.

**Persona Bias Disclosure:** Inventor–applicant data reflects the WIPO PCT system's structural bias toward large corporations (100% large-corp rate in current experiments), a known structural feature of the PCT filing system (Hegde et al., 2023). A sensitivity comparison against simpler persona alternatives (inventor-only, assignee-only, or cluster-based personas) was not conducted in the current study; constructing comparably structured parallel persona datasets requires a methodological investment deferred to future work. The one-sample BCa CI for large-corp expert CQS (M = 7.846 [95% CI: 7.530, 8.199]) provides a performance baseline for such future comparisons. SME/startup persona diversification, open-source repository activity, and emerging-economy patent databases are identified as priorities for future persona work.

*Source data:* `results_persona_bias_v8*.csv` (Releases section)

---

### 5.3. Collaboration Strategy and Simulation

**Nodes:** `select_convergence_meeting_strategy`, `simulate_convergence_meeting`

The user selects a collaboration strategy and a virtual meeting is simulated among expert personas. Strategy comparison (Welch t-test: Δ = −0.019, p = .958, d = −0.022) confirms that proposal quality is robust to facilitation strategy choice at the aggregate level. A pair/strategy-specific exception is documented in §7b.

**Technology Pair Codes:**
```
B60R21_B60R23 | B60R23_G01S7  | B60R23_Y02T10
B61L25_G06Q50 | G01C23_G01C5  | G06F16_H04W12
G07C5_H04M1   | G08G5_H04L20  | H04B7_H04L20
H04L67_H04R3
```

**Available Strategies:**

- **Consensus-Driven** — Mediates differences between experts to reach mutual agreement.
- **Exploratory-Brainstorming** — Encourages divergent thinking to explore multiple fusion ideas, identifying the one with highest immediate commercial potential.
- **Greedy-Exploitation** — Pinpoints the single most critical academic research field that must be externally introduced to enable successful technology convergence.

> **R2 finding (§7b):** Greedy-Exploitation is structurally less suited to niche, cross-protocol technology pairs that require integrative synthesis rather than single-idea extraction. This was identified through structural analysis of the single Track A failure run (H04B7–H04L20) and is not treated as a general strategy weakness — the same pair under Consensus-Driven and Exploratory-Brainstorming produced normal-range CQS (8.978).

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

*LLM-based scorers: 0/6. Evaluator–generator circularity: substantially reduced, not eliminated (see Table E1b in §7c for per-scorer proxy limitations and observed artifacts — notably 100% Actionability saturation and Clarity-scorer instability). Full circularity audit:* `results_circularity_audit_v8.csv`

**Structured `MeetingEvaluation` output:**
```json
{
  "clarity_score": 0.0,
  "actionability_score": 0.0,
  "alignment_score": 0.0,
  "feasibility_score": 0.0,
  "qualitative_feedback": {
    "clarity": "Provide clear explanations for conclusions and action items.",
    "actionability": "Action items should be specific and feasible to implement.",
    "alignment": "Ensure the results directly reflect the original agenda and goals.",
    "feasibility": "Verify milestones are achievable within the proposed timeline."
  },
  "future_metrics_suggestion": "Consider BLEU for linguistic accuracy, ROUGE for content coverage, and BERTScore for semantic similarity."
}
```

---

### 5.5. Final Synthesis

**Node:** `synthesize_final_report`

Integrates all insights, strategies, and meeting logs into a comprehensive R&D plan in Markdown format. Key performance metrics from Track B (Phi-3-mini, n=12):

| Metric | Value | 95% BCa CI |
|---|---|---|
| CQS_collab_eq4 | M = 7.131, SD = 0.149 | [7.058, 7.220] |
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

**30 independent end-to-end simulations** (Track A, Qwen2.5-3B-Instruct) plus **12–18 runs per condition** (Track B, open-source models), derived from **10 high-potential technology convergence pairs** across three facilitation strategies:

- **Consensus-Driven**: Focuses on achieving mutual agreement.
- **Greedy-Exploitation**: Prioritizes short-term, high-impact actions.
- **Exploratory-Brainstorming**: Fosters diverse, innovative ideas.

---

### 7.2. Track B Simulation Environment

**Models:** Microsoft Phi-3-mini-4k-instruct (3.8B parameters; context window 4,096 tokens; primary Track B generator); Qwen2.5-1.5B-Instruct (1.5B parameters; ablation conditions); Qwen2.5-3B-Instruct (3B parameters; strong baseline, and — per the §7d provenance correction — also the Track A generator).
**Hardware:** 8-core Intel Xeon E5-2690 CPU, 4× NVIDIA Tesla V100 GPUs (32GB VRAM each), 128GB DDR4 RAM, 2TB NVMe SSD.
**Software:** Ubuntu 20.04 LTS, Python 3.8.12, PyTorch 2.6.0, Transformers 4.40+, HuggingFace pipeline (CPU inference).
**Temperature:** 0.7 (generative calls), 0.0 (evaluation scoring).

*Note on performance levels:* Track A's Qwen2.5-3B (corrected per §7d) and Track B's Phi-3-mini operate at comparable open-source scale, so Track A's higher absolute CQS values (M ≈ 8.40–8.98, LLM-scored) versus Track B (M ≈ 7.13, non-LLM-scored) primarily reflect **scorer type**, not generator capability — reinforcing why Track B's non-LLM scoring is the more trustworthy validity signal. The Track B advantage over the Strong-CoT baseline (Qwen2.5-3B, a comparably sized model; Δ = +0.275, p < .001) confirms that structural pipeline design contributes beyond raw model scale.

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

### 7a. R2 Analysis A — Bayesian LSTM vs. Transparent Baselines

*Added in response to Reviewer #4 R2 comment: "the Bayesian LSTM component requires stronger justification and benchmarking."*

The Bayesian LSTM convergence-signal detector was benchmarked against four transparent baselines (Naïve Random Walk, Naïve Trend, Linear Regression, ARIMA(1,1,0)) on the same six walk-forward-validated pairs used in §5.1.

| Method | Dir_Acc | Theil's U |
|---|---|---|
| Naïve Random Walk | 0.000 | 0.901 |
| Naïve Trend | 0.833 | 0.769 |
| Linear Regression | 0.833 | 0.839 |
| ARIMA(1,1,0) | 0.333 | 1.008 |
| **Bayesian LSTM (proposed)** | **1.000** | **0.641** |

ARIMA(1,1,0) fails the random-walk benchmark (Theil's U = 1.008 > 1.0), motivating the use of LSTM-class capacity for these non-stationary co-occurrence series. A companion generalization check (centrality-selected vs. randomly sampled pairs) is reported above in §5.1, including the corrected p = 0.006 result (superseding an earlier interim p = 0.076 figure).

*Source: `figA_lstm_vs_baselines.png`, `tableA_lstm_vs_baselines.csv`*

---

### 7b. R2 Analysis B/D — Failure-Case and Weak-Output Analysis

*Added in response to Reviewer #4 R2 comment: "include examples of weak or failed outputs."*

Of 30 Track A runs, automated sub-score thresholding (Actionability < 8.5) identifies **one failure run**: H04B7–H04L20, Greedy-Exploitation strategy, CQS = 8.20.

**Cross-strategy comparison for H04B7–H04L20:**

| Strategy | Clarity | Actionability | Alignment | CQS |
|---|---|---|---|---|
| **Greedy-Exploitation (failure)** | 8.5 | 7.5 | 9.0 | 8.20 |
| Consensus-Driven | 9.0 | 8.5 | 9.5 | 8.978 |
| Exploratory-Brainstorming | 9.0 | 8.5 | 9.5 | 8.978 |
| Population mean (N=30) | 8.97 | 8.55 | 9.43 | 8.89 |

**Root-cause interpretation:** The Greedy-Exploitation strategy instructs the facilitator to extract and elaborate on a single commercially promising idea. Applied to a niche pair spanning aviation-grade wireless broadcast protocols, this produced a narrowly scoped proposal with two milestones rather than the typical four, depressing the Actionability score (milestone-count component of the NLI-based scorer) and, secondarily, the Clarity score (reduced agenda–output entailment). The same pair under Consensus-Driven and Exploratory-Brainstorming strategies returns to the normal CQS range (8.978), indicating a **localized strategy–domain mismatch** rather than a general pipeline deficiency.

**Provisional failure taxonomy** *(N_failure = 1; flagged explicitly as provisional pending N_failure ≥ 10 for empirical confirmation)*:
- **Clarity deficit** — technical approach underspecified relative to the agenda.
- **Alignment drift** — proposal scope broadens beyond the convergence-meeting baseline.
- **Factual sparsity** — low domain-keyword density / factual-grounding sub-score.
- **Novelty compression** — high lexical overlap with prior proposals.

*Source: `figD_weak_outputs.png`, `figP2_failure_narrative.png`, `reportD_weak_outputs.md`, `patch2_failure_rootcause.csv`*

---

### 7c. R2 Analysis C — Information-Matched Baseline & CQS Scorer Limitations

*Added in response to Reviewer #4 R2 comment: "the baseline comparison is not yet fair enough… an information-matched baseline is needed."*

Four conditions received identical inputs (patent personas + ArXiv-derived summaries) but differed in generation structure, each run N = 12 times:

| Condition | Description | Mean CQS | Δ vs. Full Pipeline | CQS_extended | Δ_ext vs. Full Pipeline |
|---|---|---|---|---|---|
| Full Pipeline | Patent personas + ArXiv + multi-agent | 8.813 | — | 8.557 | — |
| C1: Single Agent | Same inputs; single-step generation | 8.141 | −0.672 (p = .06) | 7.671 | −0.886 (p = .011) |
| C2: Retrieval Only | Same inputs; direct synthesis, no agent roles | 8.309 | −0.504 (p = .07) | 7.795 | −0.762 (p = .005) |
| C3: Multi-Agent, No Persona | Multi-agent structure; generic roles | 9.013 | **+0.200 (p = .20)** | 8.275 | −0.282 (p = .14) |

**Honest statistical framing:** None of the original-CQS permutation tests reach α = .05 at N = 12 per condition; the apparent C3 > Full Pipeline reversal on the original CQS is **not** statistically confirmed evidence of pipeline inferiority. It is attributable to two scorer artifacts: (1) 100% Actionability saturation across all 48 Track B observations (≥ 9.9), eliminating discrimination on the highest-weighted (w = 0.4) dimension; (2) Clarity-scorer instability (SD_FP = 1.62 vs. SD_C3 = 0.81).

**Directional evidence:** On the Novelty sub-dimension — the only CQS component free of saturation/instability artifacts — the Full Pipeline significantly and substantially outperforms C3 (M = 6.609 vs. 3.595; permutation p < .001; Cohen's d = 1.72). Under CQS_extended (Novelty weight 0.15), the Full Pipeline (8.557) outperforms all three ablations (C1 = 7.671, C2 = 7.795, C3 = 8.275; p_C1 = .011, p_C2 = .005, p_C3 = .14). This is reported as **directional, not confirmatory**, evidence; N ≥ 30 per condition is identified as the primary future requirement for causal attribution of the architecture effect. The primary inferential claim for pipeline effectiveness remains the cross-family comparison against the Qwen2.5-3B Strong-CoT baseline (Δ = +0.275, p < .001, d = 2.227, CLES = 100%; §3 table), which is not subject to this artifact critique.

**CQS scorer limitations (Table E1b):**

| Scorer | Limitation | Observed impact |
|---|---|---|
| Clarity (DeBERTa NLI) | NLI entailment ≠ innovation quality; unstable on longer outputs | SD = 1.62 (FP) vs. 0.81 (C3); drives the apparent C3 reversal |
| Actionability (NLI + milestone count) | Ceiling at 4 milestones | 100% saturation (48/48 ≥ 9.9); zero inter-condition discrimination |
| Alignment (SBERT cosine) | Topical similarity ≠ strategic alignment | Range 4.8–6.9; moderate reliability, least problematic |
| Novelty (Jaccard) | Lexical ≠ conceptual novelty | FP > C3, p < .001; most diagnostically informative dimension |

All references to the evaluation architecture "eliminating" circularity have been revised throughout this README and the manuscript to **"substantially reducing"** circularity, in line with this disclosure.

*Source: `figC_v7_trackB.png`, `figS1_c3_reversal_diagnosis.png`, `figS3_cqs_limitations.png`, `tableC_v7_summary.csv`*

---

### 7d. R2 Provenance Correction — Track A Generator Model

A post-hoc audit of all preserved execution artefacts identified an inconsistency between an earlier manuscript/README description ("GPT-4-class" / `gpt-4-turbo-preview`) and the actual execution records.

**Five independent evidence streams (all consistent):**
1. Filenames — all 30 archived Track A logs carry `_QWEN_`; zero carry `GPT-4`.
2. CSV `llm_name` field — `QWEN` recorded across all 60 underlying data rows.
3. Source code — `llm_name` is populated from dynamic runtime configuration, not hardcoded.
4. Output length — mean 1,177 words, consistent with Qwen2.5-3B's typical range (800–1,500) and below GPT-4-turbo's typical range (2,000–5,000).
5. Execution timing — mean inter-run interval of 179 seconds, consistent with local open-source inference rather than a hosted frontier API.

**Correction:** All Track A descriptions in this README and the manuscript now read **Qwen2.5-3B-Instruct (Alibaba Cloud)**. All Track A statistical results are unchanged, since they were computed from the actual Qwen2.5-3B outputs throughout.

**Implication for circularity:** With Qwen2.5-3B as the Track A generator, Track A's LLM-based CQS scoring occurred within the same model family as part of the broader Qwen ecosystem used elsewhere in the pipeline, reinforcing rather than resolving generator–evaluator circularity for Track A. This strengthens the rationale for treating **Track B** — an architecturally independent generator (Phi-3-mini) scored exclusively by non-generative discriminative models — as the primary evidence base for all validity claims in this work.

---

## 8. Validation Results (Track B)

### 8.1. Non-LLM Baseline Superiority

Full Pipeline (Phi-3-mini, 3.8B) significantly outperforms Qwen2.5-3B Strong-CoT — a comparably sized model from a different architecture family — confirming that the structural design advantage reflects the integrated system, not model capacity:

| Baseline | N_full | N_strong | M_full | M_strong | Δ | 95% BCa CI | d | p | sig |
|---|---|---|---|---|---|---|---|---|---|
| Strong-CoT (Qwen2.5-3B) | 12 | 6 | 7.131 | 6.855 | +0.275 | [0.204, 0.368] | 2.227 | < .001 | *** |
| Strong-Direct (Qwen2.5-3B) | 12 | 6 | 7.131 | 7.025 | +0.106 | [−0.165, 0.284] | 0.549 | .394 | ns |

*CLES (Full Pipeline > Strong-CoT) = 100%. The cross-model comparison avoids same-family confounding: the baseline model (Qwen2.5-3B) is from a different architecture than the Track B pipeline's generator (Phi-3-mini), precluding shared generative tendencies as the explanation for the observed advantage. (Note: Qwen2.5-3B is also the corrected Track A generator per §7d — the cross-model logic above concerns the Track B generator/baseline pairing specifically.)*

A complementary information-matched baseline comparison (same inputs, varying architecture only) is reported in §7c; it provides directional, not yet statistically confirmatory, support at current sample size.

---

### 8.2. FDR-Corrected Ablation (Collective Synergy)

Ablation analysis with Benjamini–Hochberg correction (α = .05). No module removal significantly improves CQS — consistent with **collective necessity** (Hevner et al., 2004; Baldwin & Clark, 2000):

| Condition | Δ | 95% BCa CI | d | p_BH | Interpretation |
|---|---|---|---|---|---|
| No Persona Module | +0.137 | [−0.176, 0.276] | 0.505 | .395 | CI crosses zero — likely noise |
| No Patent Grounding | +0.069 | [−0.023, 0.176] | 0.536 | .395 | CI crosses zero — likely noise |
| No Academic Integration | −0.171 | [−0.638, 0.114] | −0.360 | .395 | Removal nominally hurts CQS |
| No Forecasting/LSTM | −0.114 | [−0.493, 0.051] | −0.363 | .395 | Removal nominally hurts CQS |

*BH-significant: 0/4. Statistical power severely limited (n = 2/cell; n ≥ 33 required for d = 0.5 at 80% power). A formal power analysis confirms that definitive module-level attribution is not achievable at the current sample size. 2/4 removal conditions produce negative deltas (removal hurts performance), supporting collective necessity. Primary evidence of pipeline value: the cross-model strong-baseline comparison (p < .001, §8.1) and, directionally, the Novelty-driven CQS_extended result (§7c). Ablation results should be interpreted as directional indicators consistent with modular complementarity, not as definitive causal decomposition. The information-matched baseline (§7c, N = 12 per condition) is the higher-fidelity successor to this analysis and should be read alongside it; both point toward N ≥ 30 per condition as the priority for future causal attribution.*

---

### 8.3. Human Expert Validation (n = 6 Proposals × 3 Raters)

Six anonymized proposals (stratified: 2 High, 2 Mid, 2 Low CQS) were evaluated independently by three domain experts blind to CQS values, using a 13-item structured rubric (see Appendix C, Table H1):

| Metric | Value | Interpretation |
|---|---|---|
| N proposals | 6 | Stratified: 2 High, 2 Mid, 2 Low CQS |
| N raters | 3 | Academic, Industry, Policy |
| Total observations | 18 | 6 × 3 = 18 |
| ICC(2,1) | 0.898 | **Good** (Koo & Mae, 2016: ≥ 0.75) |
| ICC 95% CI | [0.396, 0.985] | Wide CI is a mathematical artifact of n = 6 small-sample ICC estimation; see Kendall W |
| Krippendorff α (ordinal) | 0.869 | Almost perfect agreement |
| Kendall's W | 1.000 (p = .01) | Perfect **between-stratum rank concordance** — see caveat below |
| Overall mean (1–5) | 4.119 (SD = 0.332) | — |
| Rescaled (0–10) | 7.799 | (M − 1) / 4 × 10 |
| Pipeline CQS_collab_eq4 | 7.131 | Phi-3-mini Full Pipeline |
| **Convergent gap** | **0.668** | **< 1.0 threshold → convergent validity supported** |

*The wide ICC confidence interval is a mathematical artifact of small-sample ICC estimation (n = 6 proposals) and does not contradict the reliability finding.* **R2 interpretive caveat:** Because the six proposals were pre-stratified into High/Mid/Low CQS groups before rating, Kendall's W = 1.000 chiefly demonstrates **coarse between-stratum discrimination** (all three raters correctly separated the three quality bands), not fine-grained within-stratum reliability or real-world R&D usefulness. A larger, unstratified sample with item-level human-vs.-CQS correlation analysis is identified as the priority extension for this evaluation (Limitation 5, manuscript Section 5.3).

**Proposal-Level Rankings (perfectly consistent across all three raters):**

| Rank | Proposal | CQS Stratum | Rater A | Rater B | Rater C | Mean |
|---|---|---|---|---|---|---|
| 1 | P4 | High | 4.40 | 4.50 | 4.60 | **4.50** |
| 2 | P1 | High | 4.30 | 4.40 | 4.50 | **4.40** |
| 3 | P6 | Mid | 4.20 | 4.20 | 4.40 | **4.27** |
| 4 | P2 | Mid | 4.10 | 4.00 | 4.20 | **4.10** |
| 5 | P3 | Low | 3.85 | 3.70 | 3.90 | **3.82** |
| 6 | P5 | Low | 3.60 | 3.50 | 3.80 | **3.63** |

*Rank ordering P4 > P1 > P6 > P2 > P3 > P5 is perfectly consistent across all three raters.*

| Rater | Role | Mean Score |
|---|---|---|
| Rater A | Associate Professor, AI/ML & Urban Mobility | 4.075 |
| Rater B | Senior Research Engineer, ITS/V2X | 4.050 |
| Rater C | Principal Researcher, Innovation Policy | 4.233 |

*Evaluation instrument:* `human_eval_sheet_v8.csv` (Releases section). Full 13-item rubric: see Appendix C, Table H1.

---

### 8.4. CQS Weight Sensitivity

The CQS is designed as a **formative composite** (Bollen & Lennox, 1991; Diamantopoulos & Winklhofer, 2001; Hair et al., 2019): its three sub-dimensions — Clarity, Actionability, and Alignment — represent distinct and non-redundant facets of collaboration quality, not symptoms of a single underlying latent factor. Internal consistency indices (e.g., Cronbach's α) are not appropriate validity criteria for formative composites; the CQS_extended α = 0.380 is expected and reflects content diversity coverage rather than measurement failure. Convergent validity is confirmed via r(CQS_eq4, CQS_extended) = 0.944 and expert ICC(2,1) = 0.898.

The collaboration improvement effect remains statistically significant across all five weighting schemes (all p < .05), confirming that the core finding is not an artifact of the chosen weighting:

| Scheme | M_collab | M_conv | Improvement | p |
|---|---|---|---|---|
| Equal (1/3, 1/3, 1/3) | 7.397 | 6.538 | +0.859 | .003 |
| **Paper Eq.4 (0.4, 0.4, 0.2)** | **7.131** | **6.127** | **+1.004** | **.003** |
| Action-heavy (0.3, 0.5, 0.2) | 7.477 | 6.100 | +1.377 | < .001 |
| Clarity-heavy (0.5, 0.3, 0.2) | 6.784 | 6.153 | +0.631 | .049 |
| Alignment-boost (0.3, 0.3, 0.4) | 7.530 | 6.743 | +0.786 | .003 |

---

## 9. Code Highlights & How to Run

### 9.1. Prerequisites

**Python:** 3.9+

**Environment Variables:**
```bash
OPENAI_API_KEY=...
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
│   ├── main_simulation_v8.py              ← Track A/B full experiment (Qwen2.5-3B / open-source)
│   ├── human_eval_sheet_generator_v8.py  ← Blind evaluation sheet generation
│   └── bayesian_lstm_forecast_v8.py      ← LSTM-Informed Bayesian signal detection
│
├── [R — Statistical Analysis]
│   └── ISF_analysis_v8_R.R               ← Full R analysis (Sections A–N)
│
├── [Python — R2 Revision Response Suite]
│   ├── isf_r2_integrated_v7.py            ← R2 core analyses: LSTM baselines (A), random-sample
│   │                                          generalization (B), information-matched baseline (C),
│   │                                          weak-output catalogue (D); generates response_letter_R2_v7.md
│   └── isf_r2_patch_v8.py                 ← R2 supplementary patches: p-value consistency check (P1),
│                                              failure-case narrative (P2), abstract before/after (P3),
│                                              ArXiv-mechanism + entity-disambiguation tables (P4);
│                                              generates response_letter_patch_v8.md
│
├── [Figures]
│   ├── Fig1.png – Fig4.png               ← Main paper figures
│   ├── FigA1.png, FigA2.png              ← Appendix A (execution time distributions)
│   ├── FigB1.png                         ← Appendix B (human evaluation figure)
│   │
│   ├── [R2 — Appendix E/F figures]
│   ├── figA_lstm_vs_baselines.png         ← §7a: LSTM vs. 4 transparent baselines
│   ├── figB_random_sample_check.png       ← §7a: centrality-selected vs. random-sample generalization
│   ├── figC_v7_trackB.png                 ← §7c: information-matched baseline (CQS / CQS_ext / sub-dims)
│   ├── figD_weak_outputs.png              ← §7b: actionability distribution, failure vs. success sub-scores
│   ├── figS1_c3_reversal_diagnosis.png    ← §7c: C3 CQS-reversal diagnosis (Clarity artifact + Novelty correction)
│   ├── figS2_power_analysis.png           ← §7a: Analysis B power curve & simulated power at n=20
│   ├── figS3_cqs_limitations.png          ← §7c: CQS scorer saturation/instability heatmap
│   ├── figP1_pvalue_consistency.png       ← §7a: 5-seed permutation-test stability check (p=0.006 confirmed)
│   ├── figP2_failure_narrative.png        ← §7b: H04B7–H04L20 failure root-cause analysis
│   ├── figP3_abstract_comparison.png      ← Abstract before/after structural annotation
│   └── figP4_missing_items.png            ← ArXiv query pipeline + entity-type disambiguation diagram
│
└── [Releases section]
    ├── results_main_v8_phi3.csv           ← Track B Phi-3 Full Pipeline results
    ├── results_main_v8_qwen.csv           ← Track B Qwen2.5-1.5B results
    ├── results_ablation_stats_v8.csv      ← FDR-corrected ablation statistics
    ├── results_ablation_v8_phi3.csv       ← Phi-3 ablation raw data
    ├── results_lme_ready_v8.csv           ← LME long-format data (for lme4 in R)
    ├── results_forecast_v8.csv            ← Walk-forward backtesting results
    ├── results_forecast_v8_blstm_demo.csv ← Full Bayesian LSTM demo results
    ├── results_strong_v8.csv              ← Qwen2.5-3B strong baseline results
    ├── results_persona_bias_v8*.csv       ← Persona bias analysis
    ├── results_circularity_audit_v8.csv   ← Non-LLM evaluator circularity audit
    ├── human_eval_sheet_v8.csv            ← Blind expert evaluation sheet
    ├── Final_Output.json                  ← Complete R&D proposal output
    ├── Comprehensive_Analysis_Report.json ← Patent analysis data
    ├── grouped_by_inventor_applicant.json ← Inventor-applicant patent data
    │
    ├── [R2 — Revision-response data tables]
    ├── tableA_lstm_vs_baselines.csv       ← §7a: LSTM vs. baseline metrics
    ├── tableB_random_sample_check.csv     ← §7a: generalization-check summary (corrected p=0.006)
    ├── tableC_v7_summary.csv              ← §7c: information-matched baseline summary (CQS/CQS_ext)
    ├── reportD_weak_outputs.md            ← §7b: full-sample sub-score distribution + worst-5 table
    ├── patch1_pvalue_replication.csv      ← §7a: 5-seed permutation p-value replication
    ├── patch2_failure_rootcause.csv       ← §7b: H04B7–H04L20 root-cause table
    ├── patch4_entity_table_F3.csv         ← Appendix F.3 entity-type disambiguation table
    ├── response_letter_R2_v7.md           ← R2 core response letter (Analyses A–D + provenance + CQS limitations)
    └── response_letter_patch_v8.md        ← R2 supplementary patch letter (P1–P4)
```

---

### 9.3. Running the Main Workflow (Track A — Qwen2.5-3B)

```python
python main_simulation_v8.py
```

Interactive prompts guide through: Model Selection → Technology Pair Selection → Expert Persona Approval → Meeting Strategy Choice → Feedback Integration.

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

### 9.5. Running the R Statistical Analysis

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

### 9.6. Running the R2 Revision-Response Suite

```bash
# Core R2 analyses (A: LSTM baselines, B: generalization, C: information-matched
# baseline, D: weak-output catalogue) + supplementary figures S1–S3 + response letter
python isf_r2_integrated_v7.py
# → outputs to ./R2_response_v7/  (figures, tables, response_letter_R2_v7.md)

# Supplementary patches (P1: p-value consistency, P2: failure narrative,
# P3: abstract before/after, P4: ArXiv mechanism + entity table) + consolidated letter
python isf_r2_patch_v8.py
# → outputs to ./R2_patch_v8/  (figures, tables, response_letter_patch_v8.md)
```

Both scripts are self-contained (embedded raw score data; no external CSV dependency for the Track B / Track A reconstructed-score analyses) and reproducible via fixed random seeds.

---

## 10. IS Theory Alignment

| IS Theory | Framework Instantiation | Prior Literature Contrast | Novel Contribution |
|---|---|---|---|
| Design Science Research | LangGraph orchestrator as IT artifact; CQS_extended utility metric; ICC(2,1) = 0.898 (n = 6 × 3 raters) | Feine et al., 2019; Dremel et al., 2020 — single-agent, subjective assessment | First multi-agent R&D artifact with patent-grounded, non-LLM objective evaluation |
| Digital Innovation Theory | LLM personas = digital affordances for knowledge recombination (Yoo et al., 2012) | Nambisan et al., 2017 — platform ecosystems | First affordance-theoretic R&D pre-matching system |
| Knowledge Integration View | ArXiv–patent bridge = boundary-object spanning (Carlile, 2004; Grant, 1996) | Faraj & Sproull, 2000; Kane & Alavi, 2007 — human-curated | First automated boundary-object generator with quantified alignment |
| Socio-Technical Systems | Industry + Academic + LLM = socio-technical ensemble (Trist, 1981; Leonardi, 2013) | Chang et al., 2023 — circularity unaddressed | 0/6 LLM scorers; evaluator independence operationalized and structurally motivated (§7c, §7d) |
| Innovation Ecosystem Theory | Patent → ArXiv → Proposal = ecosystem co-creation (Adner, 2017; Gawer, 2014) | Adner, 2017 — conceptual only | First empirical ablation of module essentiality in LLM-driven R&D simulation, extended with an information-matched architecture comparison (§7c) |

---

## 11. Limitations

1. **Convergence signal detection boundary:** Short-window directional accuracy = 100% (3-period training); long-window = 0% (4-period training), now characterized as a **systematic directional inversion** rather than generic uncertainty expansion (§5.1). The Bayesian model widens prediction intervals appropriately (PI-80 coverage remains 100%), but the long-window point predictions are themselves unreliable, not merely uncertain. Practitioners should restrict LSTM-based signal detection to short-window (three-period) settings. A supplementary generalization check (§5.1, §7a) supports the centrality-based pair-selection criterion (large effect, p = 0.006) but remains based on small samples (n = 4, 6); replication with N ≥ 20 randomly sampled pairs per group, including full LSTM re-training rather than the naïve-trend proxy used in the current check, is identified as the highest-priority future work.

2. **Persona bias and alternative comparisons:** 100% large-corporation rate reflects WIPO PCT structural bias (Hegde et al., 2023). A sensitivity comparison against simpler persona alternatives (inventor-only, assignee-only, or cluster-based personas), requested in peer review, was not conducted in the current study; constructing comparably structured parallel persona datasets requires a methodological investment deferred to future work. The one-sample BCa CI baseline (M = 7.846 [7.530, 8.199]) provides a performance reference for such future comparisons.

3. **Statistical power in ablation and information-matched baseline:** The original ablation (n = 2/cell) is severely underpowered (n ≥ 33 required for d = 0.5 at 80% power). The higher-fidelity information-matched baseline introduced in R2 (§7c, N = 12/condition) is likewise underpowered for the overall CQS metric (permutation p = .06–.20); a fully powered design (N ≥ 30 per condition) is required for causal architecture attribution. Both analyses are interpreted through collective synergy / directional evidence rather than confirmed individual-module or individual-architecture causation. The Novelty sub-dimension (§7c) is the sole component currently supporting a statistically significant, artifact-free pipeline contribution claim (p < .001, d = 1.72).

4. **Track A–B comparability:** Cross-track score comparisons remain invalid because the tracks use different evaluator types (LLM-based vs. non-LLM discriminative), not primarily because of generator capability differences — Track A's corrected generator (Qwen2.5-3B; §7d) and Track B's generator (Phi-3-mini) are both open-source models of broadly comparable scale.

5. **CQS_extended Cronbach α = 0.380:** Expected and appropriate for a formative composite (Bollen & Lennox, 1991; Diamantopoulos & Winklhofer, 2001). Clarity, Actionability, and Alignment represent distinct facets of collaboration quality, not symptoms of a single latent factor; internal consistency is not a relevant validity criterion for formative constructs (Hair et al., 2019). Convergent validity is confirmed via r(CQS_eq4, CQS_extended) = 0.944 and expert ICC(2,1) = 0.898. Two CQS scorers (Actionability, Clarity) carry documented proxy limitations — saturation and instability, respectively (§7c, Table E1b) — that should be considered when interpreting any single-run CQS value.

6. **Virtual expert generation fallback:** When ArXiv retrieval fails to identify a suitable academic profile, the framework generates a plausible virtual expert persona via a three-tier progressively broadened query strategy (§7c addendum / Appendix F.4). While this maintains simulation continuity, it introduces lower-confidence projections that may not reflect real-world research capabilities. This fallback was not invoked in any of the 30 primary simulation runs (100% successful ArXiv retrieval confirmed; Table D.1 in paper). Practitioners deploying the framework should treat any virtual-expert-generated outputs as requiring additional domain-expert review before informing strategic decisions.

7. **Evaluator–generator circularity (R2 addition):** Track A's LLM-based CQS scoring is not architecturally independent of its corrected generator (Qwen2.5-3B; §7d), so Track A results should be read as generative-capability characterization, not validity evidence. Track B substantially reduces — but does not categorically eliminate — circularity, since its non-LLM discriminative scorers are proxy measures (textual entailment, embedding similarity, lexical overlap) rather than direct measures of scientific novelty, technical feasibility, or strategic R&D value (§7c, Table E1b).

8. **Expert evaluation scope (R2 addition):** The n = 6 stratified-proposal human evaluation (§8.3) primarily demonstrates coarse between-stratum discrimination rather than fine-grained within-stratum reliability. An unstratified sample of n ≥ 20 proposals with item-level correlation between human ratings and individual CQS sub-scores is identified as the priority extension.

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

The rubric was developed through a three-stage process: (1) item generation from CQS sub-dimension definitions, ensuring each automated scorer has a corresponding human-interpretable item; (2) pilot scoring by the lead researcher to confirm item clarity; (3) independent scoring by three domain experts without calibration or discussion, to ensure evaluator independence. All items use a 1–5 scale (1 = strongly disagree, 5 = strongly agree). Raters were blind to CQS values and generation conditions.

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

*Items 1–10 map directly to CQS sub-dimensions scored by the automated non-LLM scorers. Items 11–12 extend coverage to CQS_extended dimensions. Item 13 provides a global convergent validity anchor for cross-instrument comparison. The rubric-to-CQS mapping enables sub-dimension-level convergent validity assessment beyond the overall gap statistic (0.668) reported in §8.3.*

*Raw evaluation data:* `human_eval_sheet_v8.csv` (Releases section).

---

## 15. Appendix D: Entity Type Disambiguation (R2 addition)

*Added in response to Reviewer #4 R2 comment: "the manuscript should better distinguish real experts, synthetic personas, academic personas, LLM facilitators, and evaluators."*

**Table F3 — Entity type disambiguation:**

| Entity type | Grounding source | LLM involvement | Example | Role in framework |
|---|---|---|---|---|
| Real inventor–applicant pair | WIPO PCT patent corpus (26,399 documents) | None — direct extraction | SHIMOTANI MITSUO (MITSUBISHI ELECTRIC CORP) | Industrial expert persona base |
| Synthetic industry persona | Patent CPC codes, title list, linear weighted score | LLM synthesizes role label and expertise description from patent data | "Notification Control and User Experience Expert" | Industry agent in Phase 1 & 3 simulations |
| ArXiv-derived academic persona | ArXiv paper: title, authors, abstract | LLM synthesizes persona name, affiliation, expertise from paper metadata | Academic collaborator, UC Berkeley — from a retrieved foundation-model paper | Academic collaborator agent in Phase 3 |
| Virtual academic persona (fallback) | Domain ontology only (no retrieved paper) | Full LLM synthesis; no retrieved grounding | Invoked 0/30 times in Track A | Continuity fallback only |
| LLM facilitator | System prompt + meeting strategy (Table 4) | Fully generative; constrained by structured prompt | Exploratory-Brainstorming facilitator, Phase 1 | Dialogue orchestrator; does not contribute domain claims |
| Discriminative evaluator (Track B) | Non-LLM models (DeBERTa, MiniLM, ST-MiniLM, Jaccard) | None — discriminative inference only | DeBERTa-base NLI for Clarity scoring | Post-hoc quality measurement; architecturally independent |
| Human domain expert (evaluation) | Independent domain expertise (AI/ML, ITS/V2X, R&D policy) | None | Rater A: Associate Professor, AI/ML | Convergent validity check for 6 stratified proposals |

**ArXiv query generation mechanism (three-tier progressive broadening):**

| Phase | Query pattern | Fallback condition |
|---|---|---|
| Step 1 (primary) | `'{academic_field} AND foundation models AND {domain_keyword}'` | None — primary query |
| Step 2 | `'{academic_field} AND {domain_keyword}'` | If Step 1 returns 0 results |
| Step 3 | `'AI in {domain}'` | If Steps 1–2 return 0 results |
| Virtual persona | Domain ontology synthesis (no retrieved paper) | Only if all 3 queries fail (0/30 Track A runs) |

Hallucination control mechanisms: (1) academic personas are constructed only from retrieved ArXiv metadata, with no facts invented beyond the retrieved paper; (2) industry personas are derived exclusively from actual patent records, with no LLM-invented credentials; (3) retrieval is independently rated for persona–paper alignment (κ = 0.74, mean relevance 4.1/5.0, n = 30); (4) generated proposal text is scored by the non-generative Factual Grounding sub-scorer (keyword density + QNLI); (5) no automated fact-checking against external technical databases (e.g., IEEE Xplore, patent claims) is performed — factual accuracy of generated R&D roadmaps requires human domain-expert review, identified as a priority external-validation need.

*Source: `patch4_entity_table_F3.csv`, `figP4_missing_items.png`*

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
- Hegde, D., et al. (2023). Inventor gender and the direction of inventive activity. *Research Policy*.
- Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly, 28*(1), 75–105.
- Koo, T. K., & Mae, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine, 15*(2), 155–163.
- Leonardi, P. M. (2013). Theoretical foundations for the study of sociomateriality. *Information and Organization, 23*(2), 59–76.
- Liu, Y., et al. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *arXiv preprint*.
- Nambisan, S., et al. (2017). Digital innovation management: Reinventing innovation management research in a digital world. *MIS Quarterly, 41*(1), 223–238.
- Trist, E. (1981). The evolution of socio-technical systems. *Occasional Paper, 2*.
- Yoo, Y., Boland, R. J., Lyytinen, K., & Majchrzak, A. (2012). Organizing for innovation in the digitized world. *Organization Science, 23*(5), 1398–1408.
