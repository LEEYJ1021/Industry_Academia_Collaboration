# Analysis D — Weak/Failed Output Catalogue (v7)

N=30: {'Middle': 24, 'Success (action≥9.0)': 5, 'Failure (action<8.5)': 1}

**Track A provenance:** all 30/30 logs carry '_QWEN_'.

## Full-sample sub-score distribution (N=30)

| Dimension        |   Full_N=30_Mean |   Full_N=30_SD |   Failure_N=1_Mean |   Success_N=5_Mean |   Delta_S_minus_F |
|:-----------------|-----------------:|---------------:|-------------------:|-------------------:|------------------:|
| clarity          |            8.967 |          0.225 |                8.5 |                8.9 |               0.4 |
| actionability    |            8.55  |          0.274 |                7.5 |                9   |               1.5 |
| alignment        |            9.433 |          0.173 |                9   |                9.2 |               0.2 |
| num_action_items |            5.833 |          0.791 |                6   |                5.8 |              -0.2 |

## Worst-5 by CQS (quantitative only; no text reproduction)

| run_id                                                  | convergence_pair   | collab_meeting_strategy   |   CQS_collab |   collab_actionability_score |   collab_clarity_score |   collab_alignment_score |
|:--------------------------------------------------------|:-------------------|:--------------------------|-------------:|-----------------------------:|-----------------------:|-------------------------:|
| R&D_Proposal_H04B7__H04L20_QWEN_20250620_203232_FullLog | H04B7/-H04L20      | Greedy-Exploitation       |          8.2 |                          7.5 |                    8.5 |                      9   |
| R&D_Proposal_B61L25_G06Q50_QWEN_20250620_202328_FullLog | B61L25-G06Q50      | Greedy-Exploitation       |          8.8 |                          9   |                    8.5 |                      9   |
| R&D_Proposal_B60R23_Y02T10_QWEN_20250620_200216_FullLog | B60R23-Y02T10      | Consensus-Driven          |          8.8 |                          9   |                    8.5 |                      9   |
| R&D_Proposal_G08G5__H04L20_QWEN_20250620_203107_FullLog | G08G5/-H04L20      | Greedy-Exploitation       |          8.8 |                          9   |                    8.5 |                      9   |
| R&D_Proposal_B60R23_Y02T10_QWEN_20250620_202711_FullLog | B60R23-Y02T10      | Greedy-Exploitation       |          8.9 |                          8.5 |                    9   |                      9.5 |

## Provisional failure taxonomy

*NOTE: N_failure=1 in this dataset. The four failure types above are a *provisional taxonomy* derived from one empirical case and theoretical reasoning about the CQS sub-dimensions. Empirical confirmation requires N_failure≥10.*

- **Clarity deficit**: Low clarity_score (<8.5); proposal underspecified relative to agenda.
- **Alignment drift**: Alignment below convergence baseline; proposal scope broadens excessively.
- **Factual sparsity**: Low factual-grounding sub-score; domain keyword density insufficient.
- **Novelty compression**: High Jaccard overlap with prior proposals; incremental framing.
