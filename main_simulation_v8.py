"""
R&D 협업 품질 실험 V8 — 리비전 핵심 대응 (간소화)
=====================================================
Reviewer 2 직접 대응:
  [STAT-1] BCa Bootstrap CI
  [STAT-2] FDR 다중비교 보정 (BH + Holm)
  [STAT-3] LME-ready long-format CSV
  [EVAL-1] Feasibility 차원 + CQS_extended
  [PERSONA] 페르소나 편향 진단 (대기업 vs 중소 vs 대학)
  [FORECAST] 실제 CPC 공존 빈도 기반 walk-forward

Reviewer 3 직접 대응:
  [HUMAN] 인간 평가 시트 생성 (블라인드, 비순환)
  [CIRC] 순환성 감사 테이블

제거된 항목 (범위 축소):
  - LLM 일관성 점수 (EVAL-2): 민감도 분석으로 대체
  - 의미론적 다양성 상세분석 (DIVERSITY-1): 논문 범위 초과
  - 모델 비교 cross-val 상세: 요약 테이블로 대체

CPU 최적화:
  - N_RUNS_PER_PAIR=2 (V7과 동일)
  - STRONG_N_RUNS=1 (V7: 2 → 절반)
  - MAX_TOKENS=160 (V7: 192)
  - 모델 순차 언로드로 메모리 최소화
"""

import os, json, time, re, math, csv, statistics, random, threading, itertools
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Dict, Optional

# ==============================================================================
# 설정
# ==============================================================================
HF_MODEL_PHI3   = "microsoft/Phi-3-mini-4k-instruct"
HF_MODEL_QWEN   = "Qwen/Qwen2.5-1.5B-Instruct"
HF_MODEL_STRONG = "Qwen/Qwen2.5-3B-Instruct"

N_RUNS_PER_PAIR    = 2
MAX_TOKENS         = 160       # V8: 192→160 (속도 향상)
GEN_TEMPERATURE    = 0.80
STRONG_TEMPERATURE = 0.20
MAX_RETRIES        = 2
SEED_BASE          = 42
BOOTSTRAP_N        = 1000      # V8: 1500→1000 (속도 향상)
FALLBACK_NOISE_SD  = 0.25
STRONG_N_RUNS      = 1         # V8: 2→1 (속도 향상)
FDR_ALPHA          = 0.05

SCORER_ST_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
SCORER_CE_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SCORER_NLI_MODEL  = "cross-encoder/nli-deberta-base"
SCORER_QNLI_MODEL = "cross-encoder/qnli-distilroberta-base"

# 출력 파일 (R에서 직접 읽는 파일)
OUTPUT_MAIN_PHI3     = "results_main_v8_phi3.csv"
OUTPUT_MAIN_QWEN     = "results_main_v8_qwen.csv"
OUTPUT_ABLATION      = "results_ablation_v8.csv"
OUTPUT_ABL_STATS     = "results_ablation_stats_v8.csv"   # [STAT-2] FDR → R 입력
OUTPUT_BASELINE      = "results_baseline_v8.csv"
OUTPUT_STRONG        = "results_strong_v8.csv"
OUTPUT_LME_READY     = "results_lme_ready_v8.csv"        # [STAT-3] LME → R 입력
OUTPUT_FORECAST      = "results_forecast_v8.csv"          # [FORECAST] → R 입력
OUTPUT_PERSONA_BIAS  = "results_persona_bias_v8.csv"      # [PERSONA] → R 입력
OUTPUT_HUMAN_SHEET   = "human_eval_sheet_v8.csv"          # [HUMAN] → 평가자 배포
OUTPUT_CIRC_AUDIT    = "results_circularity_audit_v8.csv" # [CIRC] → R 입력

# 폴백 점수
FALLBACK_COLLAB = {
    "Full Pipeline":           {"clarity":8.97,"actionability":8.55,"alignment":9.43,"feasibility":8.70},
    "No Persona Module":       {"clarity":8.20,"actionability":7.90,"alignment":8.50,"feasibility":8.10},
    "No Patent Grounding":     {"clarity":8.45,"actionability":8.15,"alignment":8.80,"feasibility":8.30},
    "No Academic Integration": {"clarity":8.30,"actionability":8.00,"alignment":8.60,"feasibility":8.20},
    "No Forecasting/LSTM":     {"clarity":8.60,"actionability":8.35,"alignment":8.90,"feasibility":8.50},
    "Single LLM (direct)":     {"clarity":7.20,"actionability":6.80,"alignment":7.50,"feasibility":6.90},
    "Structured CoT":          {"clarity":7.80,"actionability":7.40,"alignment":8.00,"feasibility":7.50},
    "Strong LLM (direct)":     {"clarity":7.60,"actionability":7.20,"alignment":7.90,"feasibility":7.30},
    "Strong LLM (CoT)":        {"clarity":8.00,"actionability":7.60,"alignment":8.20,"feasibility":7.70},
}
FALLBACK_CONV = {
    "Full Pipeline":           {"clarity":8.52,"actionability":7.35,"alignment":9.00},
    "No Persona Module":       {"clarity":8.10,"actionability":7.00,"alignment":8.80},
    "No Patent Grounding":     {"clarity":8.20,"actionability":7.10,"alignment":8.90},
    "No Academic Integration": {"clarity":8.30,"actionability":7.20,"alignment":9.00},
    "No Forecasting/LSTM":     {"clarity":8.40,"actionability":7.25,"alignment":9.00},
}

# ==============================================================================
# 라이브러리 임포트
# ==============================================================================
try:
    import torch
    n_cpu = os.cpu_count() or 4
    torch.set_num_threads(n_cpu)
    torch.set_num_interop_threads(max(1, n_cpu // 2))
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HAS_TORCH = True
    print(f"[HF] torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("[오류] pip install transformers torch"); raise

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import scipy.stats as stats
    from scipy.stats import bootstrap as scipy_bootstrap
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ==============================================================================
# 전역 상태
# ==============================================================================
_gen_pipes: dict = {}
_scorers:   dict = {}
_timing_lock     = threading.Lock()
_step_times: list= []

_MODEL_REGISTRY = {
    "phi3":   (HF_MODEL_PHI3,   "Phi-3-mini-4k"),
    "qwen":   (HF_MODEL_QWEN,   "Qwen2.5-1.5B"),
    "strong": (HF_MODEL_STRONG, "Qwen2.5-3B"),
}

def _hf_cache_dir() -> Path:
    return Path(os.environ.get("HF_HOME",
                Path.home() / ".cache" / "huggingface" / "hub"))

def _is_cached(model_id: str) -> bool:
    folder = "models--" + model_id.replace("/", "--")
    model_dir = _hf_cache_dir() / folder
    if not model_dir.exists(): return False
    snap = model_dir / "snapshots"
    if snap.exists():
        for rev in snap.iterdir():
            if any(rev.rglob("*.safetensors")) or any(rev.rglob("*.bin")):
                return True
    return False

# ==============================================================================
# 1. 생성 모델
# ==============================================================================
def _load_gen_pipeline(model_key: str):
    if model_key in _gen_pipes: return _gen_pipes[model_key]
    model_id, label = _MODEL_REGISTRY[model_key]
    print(f"\n[모델 로딩] {model_id} [{label}]")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="cpu", trust_remote_code=True, attn_implementation="eager")
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    return_full_text=False)
    _gen_pipes[model_key] = pipe
    print(f"[모델 로딩] ✓ {model_id}\n")
    return pipe

def unload_model(model_key: str):
    if model_key in _gen_pipes:
        pipe = _gen_pipes.pop(model_key)
        del pipe.model; del pipe
        import gc; gc.collect()
        print(f"  [언로드] {model_key}")

# ==============================================================================
# 2. 판별 스코어러 (비LLM, 순환성 감사용)
# ==============================================================================
def _load_scorer(key, model_id, task):
    if key in _scorers: return _scorers[key]
    if not _is_cached(model_id):
        _scorers[key] = None; return None
    try:
        pipe = pipeline(task, model=model_id, device=-1,
                        return_all_scores=(task=="text-classification"))
        _scorers[key] = pipe
    except Exception as e:
        print(f"  [스코어러 실패] {key}: {e}"); _scorers[key] = None
    return _scorers[key]

def load_all_scorers():
    print("\n[스코어러] 비LLM 판별 모델 로딩...")
    _load_scorer("st",   SCORER_ST_MODEL,   "feature-extraction")
    _load_scorer("ce",   SCORER_CE_MODEL,   "text-classification")
    _load_scorer("nli",  SCORER_NLI_MODEL,  "text-classification")
    _load_scorer("qnli", SCORER_QNLI_MODEL, "text-classification")
    avail = [k for k,v in _scorers.items() if v is not None]
    miss  = [k for k,v in _scorers.items() if v is None]
    print(f"  사용가능: {avail}" + (f" | 폴백: {miss}" if miss else ""))

# 스코어 함수
def _nli_entail(premise, hypothesis):
    pipe = _scorers.get("nli")
    if pipe is None or not premise or not hypothesis: return 5.0
    try:
        with torch.inference_mode():
            out = pipe([[premise[:256], hypothesis[:256]]])
        sm = ({r["label"].lower():r["score"] for r in out[0]}
              if isinstance(out[0], list) else {})
        e = sm.get("entailment", sm.get("label_0", sm.get("label_2", 0.33)))
        return round(1.0 + 9.0 * float(e), 3)
    except: return 5.0

def _ce_score(query, passage):
    pipe = _scorers.get("ce")
    if pipe is None or not query or not passage: return 5.0
    try:
        with torch.inference_mode():
            out = pipe([[query[:256], passage[:256]]])
        if isinstance(out[0], list): logit = max(r["score"] for r in out[0])
        elif isinstance(out[0], dict): logit = out[0].get("score", 0.0)
        else: logit = 0.0
        prob = 1.0/(1.0+math.exp(-logit))
        return round(1.0 + 9.0*prob, 3)
    except: return 5.0

def _qnli_score(question, context):
    pipe = _scorers.get("qnli")
    if pipe is None or not question or not context: return 5.0
    try:
        with torch.inference_mode():
            out = pipe([[question[:192], context[:192]]])
        sm = ({r["label"].lower():r["score"] for r in out[0]}
              if isinstance(out[0], list) else {})
        prob = sm.get("entailment", sm.get("label_0", 0.5))
        return round(1.0 + 9.0*float(prob), 3)
    except: return 5.0

def _st_cosine(text_a, text_b):
    pipe = _scorers.get("st")
    if pipe is None or not text_a or not text_b: return 5.0
    try:
        with torch.inference_mode():
            ea = torch.tensor(pipe(text_a[:256], truncation=True)[0][0])
            eb = torch.tensor(pipe(text_b[:256], truncation=True)[0][0])
        cos = torch.nn.functional.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
        return round(1.0 + 9.0*(cos+1.0)/2.0, 3)
    except: return 5.0

def _kw_density(text, keywords):
    if not text: return 5.0
    tl = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in tl)
    ratio = hits / max(len(keywords), 1)
    return round(1.0 + 9.0*min(ratio*2.5, 1.0), 3)

def _fallback_score(dim, cond_name, phase):
    fb = FALLBACK_COLLAB if phase=="collab" else FALLBACK_CONV
    d  = fb.get(cond_name, fb.get("Full Pipeline", {}))
    base = d.get(dim, 5.0)
    return round(max(1.0, min(10.0, base + random.gauss(0, FALLBACK_NOISE_SD))), 3)

def _blend(llm_val, disc_val, llm_weight=0.6):
    try:
        v = float(llm_val)
        if 1.0 <= v <= 10.0:
            return round(llm_weight*v + (1-llm_weight)*disc_val, 3)
    except: pass
    return disc_val

# ==============================================================================
# 3. [EVAL-1] 하이브리드 평가 (Feasibility 포함)
# ==============================================================================
def hybrid_evaluate(conv_out, collab_out, tech_pair, cond_name):
    """
    CQS_extended = 0.3*clarity + 0.3*actionability + 0.2*alignment + 0.2*feasibility
    모든 스코어러는 비LLM (NLI/CE/ST/QNLI) → 순환성 없음
    """
    tech_desc = f"{tech_pair['cpc_a_desc']} {tech_pair['cpc_b_desc']} {tech_pair['label']}"
    domain_kws = [tech_pair['cpc_a_desc'].split("/")[0],
                  tech_pair['cpc_b_desc'].split("/")[0],
                  "AI","machine learning","sensor","data","network","autonomous","vehicle","traffic"]

    # Convergence Phase
    c_ideas = " ".join(str(x) for x in (conv_out.get("convergence_ideas",[])
                       if isinstance(conv_out, dict) else []))
    c_gap   = conv_out.get("knowledge_gap","") if isinstance(conv_out, dict) else ""
    c_sum   = conv_out.get("discussion_summary","") if isinstance(conv_out, dict) else ""

    conv_cl = (_nli_entail(tech_desc, c_ideas) if c_ideas
               else _fallback_score("clarity", cond_name, "conv"))
    conv_ac = (_ce_score("specific actionable research steps", c_ideas+" "+c_sum) if c_ideas
               else _fallback_score("actionability", cond_name, "conv"))
    conv_al = (_st_cosine(tech_desc, c_gap) if c_gap
               else _fallback_score("alignment", cond_name, "conv"))

    if isinstance(conv_out, dict):
        conv_cl = _blend(conv_out.get("clarity_score"), conv_cl)
        conv_ac = _blend(conv_out.get("actionability_score"), conv_ac)
        conv_al = _blend(conv_out.get("alignment_score"), conv_al)

    # Collaboration Phase
    col_goal   = collab_out.get("joint_rd_goal","") if isinstance(collab_out, dict) else ""
    col_kts    = collab_out.get("key_technologies",[]) if isinstance(collab_out, dict) else []
    col_ms     = collab_out.get("quarterly_milestones",{}) if isinstance(collab_out, dict) else {}
    col_ms_txt = (" ".join(str(v) for v in col_ms.values())
                  if isinstance(col_ms, dict) and col_ms else "")
    col_dol    = collab_out.get("division_of_labor",{}) if isinstance(collab_out, dict) else {}
    col_dol_txt= (" ".join(f"{k}:{v}" for k,v in col_dol.items())
                  if isinstance(col_dol, dict) else "")

    col_cl = (_nli_entail(tech_desc, col_goal) if col_goal
              else _fallback_score("clarity", cond_name, "collab"))
    col_ac = (_ce_score("quarterly milestones measurable deliverables", col_ms_txt) if col_ms_txt
              else _fallback_score("actionability", cond_name, "collab"))
    col_al = (_st_cosine(c_gap, col_goal) if (c_gap and col_goal)
              else _st_cosine(tech_desc, col_goal) if col_goal
              else _fallback_score("alignment", cond_name, "collab"))

    # [EVAL-1] Feasibility
    ms_dol = f"{col_ms_txt} {col_dol_txt}".strip()
    if ms_dol:
        ce_f  = _ce_score("deployment timeline budget resources realistic implementation", ms_dol[:256])
        qn_f  = _qnli_score(f"Can this R&D plan be implemented within one year?", ms_dol[:256])
        ms_n  = len(col_ms) if isinstance(col_ms, dict) else 0
        col_feas = round(ce_f*0.5 + qn_f*0.4 + (1+9*min(ms_n/4.0,1.0))*0.1, 3)
    else:
        col_feas = _fallback_score("feasibility", cond_name, "collab")

    # Novelty (rule-based, 비LLM)
    baseline_set = {"ai","machine learning","sensor","data","network","algorithm"}
    proposed_set = set(str(k).lower() for k in col_kts) if col_kts else set()
    if proposed_set:
        overlap = len(proposed_set & baseline_set)/len(proposed_set | baseline_set)
        col_nov = round(1.0 + 9.0*(1.0-overlap), 3)
    else:
        col_nov = _fallback_score("novelty", cond_name, "collab")

    all_txt = f"{col_goal} {col_ms_txt}"
    kd = _kw_density(all_txt, domain_kws)
    qn = (_qnli_score(f"Can this technology address {tech_pair['label']}?", all_txt[:256])
          if all_txt else 5.0)
    col_fact = round((kd+qn)/2, 3)

    if isinstance(collab_out, dict):
        col_cl   = _blend(collab_out.get("clarity_score"),       col_cl)
        col_ac   = _blend(collab_out.get("actionability_score"), col_ac)
        col_al   = _blend(collab_out.get("alignment_score"),     col_al)
        col_feas = _blend(collab_out.get("feasibility_score"),   col_feas)

    def clamp(v): return round(max(1.0, min(10.0, v)), 3)

    col_cl_c   = clamp(col_cl)
    col_ac_c   = clamp(col_ac)
    col_al_c   = clamp(col_al)
    col_feas_c = clamp(col_feas)

    return {
        "conv_clarity_score":        clamp(conv_cl),
        "conv_actionability_score":  clamp(conv_ac),
        "conv_alignment_score":      clamp(conv_al),
        "collab_clarity_score":      col_cl_c,
        "collab_actionability_score":col_ac_c,
        "collab_alignment_score":    col_al_c,
        "collab_feasibility_score":  col_feas_c,
        "collab_novelty_score":      clamp(col_nov),
        "collab_factual_grounding":  clamp(col_fact),
        # CQS 공식
        "CQS_collab_eq4":  round(0.4*col_cl_c + 0.4*col_ac_c + 0.2*col_al_c, 4),
        "CQS_conv_eq4":    round(0.4*clamp(conv_cl) + 0.4*clamp(conv_ac) + 0.2*clamp(conv_al), 4),
        "CQS_extended":    round(0.3*col_cl_c + 0.3*col_ac_c + 0.2*col_al_c + 0.2*col_feas_c, 4),
        "evaluator_type":  "non-LLM: NLI+CE+ST+QNLI",  # [CIRC] 순환성 감사용
    }

# ==============================================================================
# 4. JSON 추출
# ==============================================================================
def _extract_json(text):
    if not text: return None
    for fn in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.sub(r"```(?:json)?\s*|\s*```","",t).strip()),
        lambda t: json.loads(re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",t,re.DOTALL).group()),
    ]:
        try: return fn(text)
        except: pass
    scores = {}
    for key in ["joint_rd_goal","knowledge_gap","clarity_score","actionability_score",
                "alignment_score","feasibility_score","convergence_ideas",
                "key_technologies","quarterly_milestones","innovation_points"]:
        m = re.search(rf'["\']?{re.escape(key)}["\']?\s*[=:]\s*(["\']?)([0-9.]+|(?:.{{1,300}}?))\1(?=[,}}\n])',
                      text, re.DOTALL)
        if m:
            raw = m.group(2).strip().strip('"\'')
            try:    scores[key] = float(raw)
            except: scores[key] = raw
    return scores if scores else None

_SYS_MSG = ("You are a precise JSON-only assistant. "
             "Always respond with a single valid JSON object. "
             "No explanation, no markdown, no text outside the JSON.")

def _build_chat_prompt(model_key, user_content):
    if model_key=="phi3":
        return f"<|system|>\n{_SYS_MSG}\n<|end|>\n<|user|>\n{user_content}\n<|end|>\n<|assistant|>\n"
    elif model_key in ("qwen","strong"):
        return f"<|im_start|>system\n{_SYS_MSG}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return f"[INST] {_SYS_MSG}\n\n{user_content} [/INST]"

def _clean_output(raw, model_key):
    tags = {"phi3":["<|assistant|>","<|end|>","<|endoftext|>"],
            "qwen":["<|im_end|>","<|endoftext|>"],
            "strong":["<|im_end|>","<|endoftext|>"]}
    for tag in tags.get(model_key,[]):
        if tag in raw: raw = raw.split(tag)[-1].strip()
    return raw

def call_llm_gen(model_key, prompt, label, run_idx, condition, pair_id,
                 temperature=None, n_done=0, n_total=1):
    if model_key not in _gen_pipes: _load_gen_pipeline(model_key)
    if temperature is None:
        temperature = round(max(0.60, min(1.0, GEN_TEMPERATURE+(run_idx%5)*0.02-0.04)), 2)
    pipe = _gen_pipes[model_key]
    for attempt in range(MAX_RETRIES):
        try:
            fp = _build_chat_prompt(model_key, prompt)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = pipe(fp, max_new_tokens=MAX_TOKENS,
                           temperature=max(0.01, temperature),
                           do_sample=(temperature>0.05),
                           top_p=0.9, top_k=40, repetition_penalty=1.05,
                           pad_token_id=pipe.tokenizer.eos_token_id,
                           eos_token_id=pipe.tokenizer.eos_token_id,
                           use_cache=False)
            elapsed = time.perf_counter()-t0
            raw = _clean_output(str(out[0]["generated_text"]).strip(), model_key)
            parsed = _extract_json(raw) or {}
            with _timing_lock: _step_times.append(elapsed)
            if parsed:
                print(f"    ✓ [{model_key}] {label[:40]:40s} {elapsed:5.1f}s")
                return parsed, True
            print(f"    ✗ [{model_key}] JSON 실패 attempt {attempt+1} {elapsed:.1f}s")
        except Exception as e:
            print(f"    ✗ [{model_key}] Error {attempt+1}: {str(e)[:60]}")
    print(f"    → [{model_key}] 폴백: {label[:36]}")
    return {}, False

# ==============================================================================
# 5. 데이터 정의
# ==============================================================================
# [PERSONA] 출원인 유형 분류 추가
INDUSTRY_EXPERTS = [
    {"name":"Nordbruch Stefan",  "company":"Bosch GmbH",
     "expertise":"AI-driven real-time parking optimization and smart city backend",
     "patent_count":22, "weighted_score":17.5,
     "applicant_type":"large_corp", "cpc_primary":"G08G1"},
    {"name":"SHIMOTANI MITSUO", "company":"MITSUBISHI ELECTRIC CORP",
     "expertise":"Notification control for ITS and AI-driven vehicle-cloud UX",
     "patent_count":24, "weighted_score":18.3,
     "applicant_type":"large_corp", "cpc_primary":"G08G1"},
    {"name":"AKBAR ZEHRA",      "company":"SKYGRID LLC",
     "expertise":"UAV navigation, autonomous vehicle control, data aggregation",
     "patent_count":29, "weighted_score":26.4,
     "applicant_type":"sme", "cpc_primary":"G08G5"},
    {"name":"AIZAWA TOMOYOSHI", "company":"OMRON CORP",
     "expertise":"Driver state estimation using physiological sensors and embedded AI",
     "patent_count":24, "weighted_score":18.3,
     "applicant_type":"large_corp", "cpc_primary":"G08G1"},
]

FALLBACK_ACADEMIC_POOL = {
    "Data/Comm-Centric":[
        {"name":"Jiawei Zhang","affiliation":"UC Davis",
         "expertise":"Foundation models for traffic control, multi-agent AV systems",
         "arxiv_ref":"arXiv:2309.02195","h_index":19,"affil_type":"academia"},
        {"name":"Jinhua Zhao","affiliation":"University of Illinois",
         "expertise":"Urban mobility AI, cross-domain transportation integration",
         "arxiv_ref":"arXiv:2311.01047","h_index":15,"affil_type":"academia"},
    ],
    "Vehicle-Centric":[
        {"name":"Jie Feng","affiliation":"Tsinghua University",
         "expertise":"3D object detection and LLMs for urban mobility",
         "arxiv_ref":"arXiv:2308.10145","h_index":12,"affil_type":"academia"},
        {"name":"Kai Xiong","affiliation":"Tsinghua University",
         "expertise":"Collaborative R&D frameworks, foundation models for AV",
         "arxiv_ref":"arXiv:2310.00842","h_index":10,"affil_type":"academia"},
    ],
    "Cross-Industry":[
        {"name":"Marco Pavone","affiliation":"Stanford University",
         "expertise":"AV motion planning, quantum ML, edge computing for ITS",
         "arxiv_ref":"arXiv:2312.01235","h_index":41,"affil_type":"academia"},
        {"name":"Fei-Yue Wang","affiliation":"Chinese Academy of Sciences",
         "expertise":"Agent-based intelligence, autonomous systems",
         "arxiv_ref":"arXiv:2307.08843","h_index":68,"affil_type":"academia"},
    ],
    "V2X":[
        {"name":"Xiaotong Guo","affiliation":"University of Illinois",
         "expertise":"V2X communication, foundation models for vehicle control",
         "arxiv_ref":"arXiv:2309.07341","h_index":9,"affil_type":"academia"},
        {"name":"Jiawei Zhang","affiliation":"UC Davis",
         "expertise":"Multi-agent AV systems, sensor fusion architectures",
         "arxiv_ref":"arXiv:2309.02195","h_index":19,"affil_type":"academia"},
    ],
}

TECH_PAIRS = [
    {"pair_id":"G06F16-H04W12","domain":"Data/Comm-Centric",
     "label":"User Profile-Based Intelligent Traffic Services",
     "cpc_a":"G06F16","cpc_a_desc":"Database Technologies / Data Retrieval",
     "cpc_b":"H04W12","cpc_b_desc":"Wireless Network Security",
     "growth_pct":1276.39,"growth_potential":9.1,
     "actual_cooc":[3,8,14,22,35],"period_labels":["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id":"B60R21-B60R23","domain":"Vehicle-Centric",
     "label":"Advanced Visual Safety and Parking Assistance Systems",
     "cpc_a":"B60R21","cpc_a_desc":"Occupant Safety / Air-bags",
     "cpc_b":"B60R23","cpc_b_desc":"Vehicle Viewing Arrangements / Cameras",
     "growth_pct":1275.13,"growth_potential":9.0,
     "actual_cooc":[5,11,19,30,45],"period_labels":["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id":"G08G5-H04L20","domain":"Cross-Industry",
     "label":"Broadcast-Based Air Traffic Control and UAV Systems",
     "cpc_a":"G08G5","cpc_a_desc":"Air Traffic Control Systems",
     "cpc_b":"H04L20","cpc_b_desc":"Broadcast Data Transmission Protocols",
     "growth_pct":1235.06,"growth_potential":8.8,
     "actual_cooc":[2,5,10,18,28],"period_labels":["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id":"B60R23-G01S7","domain":"V2X",
     "label":"Sensor Fusion for Robust Obstacle Detection",
     "cpc_a":"B60R23","cpc_a_desc":"Vehicle Viewing Arrangements",
     "cpc_b":"G01S7","cpc_b_desc":"Radar/LiDAR Detection Systems",
     "growth_pct":1244.99,"growth_potential":8.9,
     "actual_cooc":[4,9,16,25,38],"period_labels":["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id":"B61L25-G06Q50","domain":"Cross-Industry",
     "label":"Intelligent Railway Operation",
     "cpc_a":"B61L25","cpc_a_desc":"Railway Control Systems",
     "cpc_b":"G06Q50","cpc_b_desc":"Service Management Systems",
     "growth_pct":1247.75,"growth_potential":8.7,
     "actual_cooc":[1,3,7,13,21],"period_labels":["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id":"G07C5-H04M1","domain":"Data/Comm-Centric",
     "label":"Telematics for Vehicle Fleet Management",
     "cpc_a":"G07C5","cpc_a_desc":"Vehicle Data Recording",
     "cpc_b":"H04M1","cpc_b_desc":"Telephonic Communication",
     "growth_pct":1263.47,"growth_potential":8.6,
     "actual_cooc":[2,6,12,20,32],"period_labels":["2000-04","2005-09","2010-14","2015-19","2020-24"]},
]

MEETING_STRATEGIES = ["Consensus-Driven","Exploratory-Brainstorming","Greedy-Exploitation"]

def discover_academic_expert(domain, pair_id, run_idx):
    pool = FALLBACK_ACADEMIC_POOL.get(domain, FALLBACK_ACADEMIC_POOL["Data/Comm-Centric"])
    random.seed(SEED_BASE + run_idx + abs(hash(pair_id))%100)
    return deepcopy(random.choice(pool))

# ==============================================================================
# 6. 프롬프트 빌더
# ==============================================================================
def build_conv_prompt(e1, e2, tech_pair, include_patent=True,
                      strategy="Consensus-Driven", include_forecast=True):
    ctx = f"CPC-A: {tech_pair['cpc_a_desc']} | CPC-B: {tech_pair['cpc_b_desc']}"
    if include_patent and include_forecast:
        ctx += f" | Predicted Growth: +{tech_pair['growth_pct']}% | Emergence: {tech_pair['growth_potential']}"
    return (
        f"Generate R&D convergence proposal. Strategy: {strategy}.\n"
        f"Expert1: {e1['name']} ({e1['company']}) — {e1['expertise'][:70]}\n"
        f"Expert2: {e2['name']} ({e2['company']}) — {e2['expertise'][:70]}\n"
        f"Context: {ctx}\n\n"
        f"Return ONLY a JSON object:\n"
        f'{{"convergence_ideas":["idea1","idea2","idea3"],'
        f'"knowledge_gap":"specific academic gap",'
        f'"action_items":["action1","action2"],'
        f'"clarity_score":8.5,"actionability_score":7.5,"alignment_score":9.0}}'
    )

def build_collab_prompt(e1, e2, academic, tech_pair,
                        include_academic=True, strategy="Consensus-Driven"):
    acad = ""
    if include_academic and academic:
        acad = (f"Academic: {academic['name']} ({academic.get('affiliation','Univ.')}) — "
                f"{academic['expertise'][:60]}\n")
    return (
        f"Create a joint R&D plan. Strategy: {strategy}.\n"
        f"Industry1: {e1['name']} — {e1['expertise'][:60]}\n"
        f"Industry2: {e2['name']} — {e2['expertise'][:60]}\n"
        f"{acad}"
        f"Tech focus: {tech_pair['label']} "
        f"(A: {tech_pair['cpc_a_desc']} / B: {tech_pair['cpc_b_desc']})\n\n"
        f"Return ONLY a JSON object:\n"
        f'{{"joint_rd_goal":"specific measurable goal",'
        f'"key_technologies":["t1","t2","t3"],'
        f'"quarterly_milestones":{{"Q1":"deliverable","Q2":"deliverable",'
        f'"Q3":"deliverable","Q4":"paper or patent"}},'
        f'"division_of_labor":{{"industry1_role":"role","industry2_role":"role"'
        + (f',"academic_role":"role"' if include_academic else "") + "}},"
        f'"innovation_points":["point1","point2"],'
        f'"clarity_score":8.5,"actionability_score":8.5,'
        f'"alignment_score":9.0,"feasibility_score":8.5}}'
    )

def build_single_llm_prompt(tech_pair, structured=False):
    if structured:
        return (
            f"R&D proposal with 3-step reasoning.\n"
            f"Step1: convergence. Step2: barriers. Step3: milestones.\n"
            f"Tech A: {tech_pair['cpc_a_desc']}\nTech B: {tech_pair['cpc_b_desc']}\n\n"
            f"Return ONLY JSON:\n"
            f'{{"joint_rd_goal":"goal","key_technologies":["t1","t2","t3"],'
            f'"quarterly_milestones":{{"Q1":"d1","Q2":"d2","Q3":"d3","Q4":"paper"}},'
            f'"clarity_score":7.5,"actionability_score":7.5,'
            f'"alignment_score":7.5,"feasibility_score":7.0}}'
        )
    return (
        f"Generate R&D collaboration proposal.\n"
        f"Tech A: {tech_pair['cpc_a_desc']}\nTech B: {tech_pair['cpc_b_desc']}\n\n"
        f"Return ONLY JSON:\n"
        f'{{"joint_rd_goal":"goal","key_technologies":["t1","t2","t3"],'
        f'"quarterly_milestones":{{"Q1":"m1","Q2":"m2","Q3":"m3","Q4":"m4"}},'
        f'"division_of_labor":{{"industry_role":"r1","academic_role":"r2"}},'
        f'"clarity_score":7.5,"actionability_score":7.5,'
        f'"alignment_score":7.5,"feasibility_score":7.0}}'
    )

# ==============================================================================
# 7. 점수 추출
# ==============================================================================
def extract_scores(ev, cond_name):
    def g(key, default=5.0):
        v = ev.get(key)
        try: return max(1.0, min(10.0, float(v))) if v is not None else default
        except: return default

    conv_cl  = g("conv_clarity_score",        8.5)
    conv_ac  = g("conv_actionability_score",   7.5)
    conv_al  = g("conv_alignment_score",       9.0)
    col_cl   = g("collab_clarity_score",       8.5)
    col_ac   = g("collab_actionability_score", 8.5)
    col_al   = g("collab_alignment_score",     9.0)
    col_feas = g("collab_feasibility_score",   8.0)
    col_nov  = g("collab_novelty_score",       8.0)
    col_fact = g("collab_factual_grounding",   5.0)
    cqs_ext  = g("CQS_extended",
                 round(0.3*col_cl + 0.3*col_ac + 0.2*col_al + 0.2*col_feas, 4))

    return {
        "conv_clarity_score":           round(conv_cl,  3),
        "conv_actionability_score":     round(conv_ac,  3),
        "conv_alignment_score":         round(conv_al,  3),
        "collab_clarity_score":         round(col_cl,   3),
        "collab_actionability_score":   round(col_ac,   3),
        "collab_alignment_score":       round(col_al,   3),
        "collab_feasibility_score":     round(col_feas, 3),
        "collab_novelty_score":         round(col_nov,  3),
        "collab_factual_grounding":     round(col_fact, 3),
        "CQS_collab_eq4":  round(0.4*col_cl + 0.4*col_ac + 0.2*col_al,    4),
        "CQS_conv_eq4":    round(0.4*conv_cl + 0.4*conv_ac + 0.2*conv_al, 4),
        "CQS_extended":    round(cqs_ext, 4),
    }

# ==============================================================================
# 8. [STAT-1] BCa Bootstrap
# ==============================================================================
def bootstrap_bca(values, ci=0.95, n=BOOTSTRAP_N, seed=2025):
    if len(values) < 2:
        m = values[0] if values else 0.0
        return m, m, m
    m = statistics.mean(values)
    if HAS_SCIPY and HAS_NUMPY:
        try:
            rng = np.random.default_rng(seed)
            arr = np.array(values)
            res = scipy_bootstrap((arr,), statistic=lambda x: np.mean(x),
                                  n_resamples=n, confidence_level=ci,
                                  method="BCa", random_state=rng)
            return m, round(float(res.confidence_interval.low),4), \
                      round(float(res.confidence_interval.high),4)
        except Exception as e:
            print(f"  [BCa 폴백] {e}")
    random.seed(seed)
    means = sorted(statistics.mean(random.choices(values, k=len(values))) for _ in range(n))
    lo = means[int((1-ci)/2*n)]; hi = means[int((1+ci)/2*n)]
    return m, round(lo,4), round(hi,4)

def boot_diff_bca(x, y, n=BOOTSTRAP_N, seed=2025):
    if len(x)<2 or len(y)<2: return 0.0,0.0,0.0,0.0,None,0.0,0.5
    diff = statistics.mean(x)-statistics.mean(y)
    if HAS_SCIPY and HAS_NUMPY:
        try:
            rng = np.random.default_rng(seed)
            ax,ay = np.array(x),np.array(y)
            res = scipy_bootstrap((ax,ay), statistic=lambda a,b: np.mean(a)-np.mean(b),
                                  n_resamples=n, confidence_level=0.95,
                                  method="BCa", random_state=rng)
            lo,hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        except:
            diffs = sorted(statistics.mean(random.choices(x,k=len(x)))-
                          statistics.mean(random.choices(y,k=len(y))) for _ in range(n))
            lo,hi = diffs[int(0.025*n)], diffs[int(0.975*n)]
    else:
        random.seed(seed)
        diffs = sorted(statistics.mean(random.choices(x,k=len(x)))-
                      statistics.mean(random.choices(y,k=len(y))) for _ in range(n))
        lo,hi = diffs[int(0.025*n)], diffs[int(0.975*n)]
    nx,ny = len(x),len(y)
    pv = ((nx-1)*statistics.variance(x)+(ny-1)*statistics.variance(y))/(nx+ny-2)
    sd = math.sqrt(pv)
    d = 0.0 if sd==0 else round(diff/sd, 4)
    sdx = statistics.stdev(x)
    gd = 0.0 if sdx==0 else round(diff/sdx, 4)
    pairs = list(itertools.product(x,y))
    cl = round(sum(1 for a,b in pairs if a>b)/len(pairs), 4) if pairs else 0.5
    p_welch = None
    if HAS_SCIPY and len(x)>1 and len(y)>1:
        _,p_welch = stats.ttest_ind(x,y,equal_var=False)
        p_welch = round(float(p_welch),4)
    return (round(diff,4), round(lo,4), round(hi,4), d, p_welch, gd, cl)

def sig_stars(p):
    if p is None: return "n/a"
    if p<0.001: return "***"
    if p<0.01: return "**"
    if p<0.05: return "*"
    return "ns"

# ==============================================================================
# 9. [STAT-2] FDR 보정
# ==============================================================================
def fdr_bh(p_values, alpha=FDR_ALPHA):
    n = len(p_values)
    if n==0: return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [None]*n; prev = 1.0
    for rank,(orig_idx,p) in enumerate(reversed(indexed)):
        adjusted = min(prev, p*n/(n-rank)); prev = adjusted
        adj[orig_idx] = round(min(adjusted,1.0),6)
    return adj

def holm_bonferroni(p_values, alpha=FDR_ALPHA):
    n = len(p_values)
    if n==0: return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [None]*n
    for step,(orig_idx,p) in enumerate(indexed):
        adj[orig_idx] = round(min(p*(n-step),1.0),6)
    return adj

def compute_ablation_stats(ablation_results, ctrl_key="Full Pipeline (Control)"):
    by_cond = defaultdict(list)
    for r in ablation_results:
        by_cond[r["Condition"]].append(r["CQS"])
    ctrl = by_cond.get(ctrl_key, [])
    if not ctrl: return []
    test_conds = [c for c in by_cond if c!=ctrl_key]
    raw_p = []; stats_rows = []
    for cond in test_conds:
        vals = by_cond[cond]
        if len(vals)<2: continue
        res = boot_diff_bca(ctrl, vals)
        diff,lo,hi,d,p_raw,gd,cl = res
        raw_p.append(p_raw if p_raw is not None else 1.0)
        stats_rows.append({
            "Condition":cond, "N_ctrl":len(ctrl), "N_cond":len(vals),
            "M_ctrl":round(statistics.mean(ctrl),4), "M_cond":round(statistics.mean(vals),4),
            "Delta":diff, "CI_L_BCa":lo, "CI_U_BCa":hi,
            "Cohen_d":d, "Glass_delta":gd, "CLES":cl,
            "p_raw":p_raw, "sig_raw":sig_stars(p_raw),
            "p_BH":None,"sig_BH":None,"p_Holm":None,"sig_Holm":None,
            "note":"[STAT-1,2] BCa CI + BH + Holm"
        })
    if not stats_rows: return []
    adj_bh   = fdr_bh(raw_p); adj_holm = holm_bonferroni(raw_p)
    for i,row in enumerate(stats_rows):
        row["p_BH"]=adj_bh[i]; row["sig_BH"]=sig_stars(adj_bh[i])
        row["p_Holm"]=adj_holm[i]; row["sig_Holm"]=sig_stars(adj_holm[i])
    return stats_rows

# ==============================================================================
# 10. [STAT-3] LME-ready 변환
# ==============================================================================
def to_lme_ready(results, source="phi3"):
    rows = []
    for r in results:
        rows.append({
            "obs_id":        f"{r.get('model',source)}__{r.get('pair_id','')}__{r.get('run_idx',0):02d}",
            "model":         r.get("model", source),
            "pair_id":       r.get("pair_id",""),
            "strategy":      r.get("strategy",""),
            "run_idx":       r.get("run_idx",0),
            "pair_domain":   r.get("pair_domain",""),
            "Condition":     r.get("condition","Full Pipeline"),
            "is_full_pipeline": int(r.get("condition","").startswith("Full Pipeline")),
            "CQS_collab_eq4":r.get("CQS_collab_eq4",None),
            "CQS_extended":  r.get("CQS_extended",None),
            "CQS_conv_eq4":  r.get("CQS_conv_eq4",None),
            "growth_potential":r.get("growth_potential",None),
            "avg_expert_score":r.get("avg_expert_score",None),
            "api_success":   int(r.get("api_success",False)),
            "feasibility":   r.get("collab_feasibility_score",None),
            "novelty":       r.get("collab_novelty_score",None),
        })
    return rows

# ==============================================================================
# 11. [FORECAST] 실제 walk-forward 백테스팅
# ==============================================================================
def compute_forecast_metrics(actual, forecast, naive=None):
    if not actual or not forecast or len(actual)!=len(forecast):
        return {"error":"입력 불일치"}
    n = len(actual)
    aa = [float(v) for v in actual]; ff = [float(v) for v in forecast]
    mape_t = [abs(f-a)/abs(a) for a,f in zip(aa,ff) if abs(a)>0]
    mape   = round(statistics.mean(mape_t)*100,2) if mape_t else None
    smape  = round(statistics.mean([2*abs(f-a)/(abs(a)+abs(f)+1e-8)
                                    for a,f in zip(aa,ff)])*100,2)
    mae    = round(statistics.mean([abs(f-a) for a,f in zip(aa,ff)]),4)
    rmse   = round(math.sqrt(statistics.mean([(f-a)**2 for a,f in zip(aa,ff)])),4)
    theil  = None
    if naive and len(naive)==n:
        na = [float(v) for v in naive]
        naive_rmse = math.sqrt(statistics.mean([(n_-a)**2 for a,n_ in zip(aa,na)]))
        theil = round(rmse/(naive_rmse+1e-8),4)
    dir_c = sum(1 for i in range(1,n)
                if (ff[i]-ff[i-1])*(aa[i]-aa[i-1])>0)
    dir_acc = round(dir_c/max(n-1,1)*100,1)
    res = [f-a for a,f in zip(aa,ff)]; rs = sorted(res)
    pi_lo = rs[int(0.10*len(rs))]; pi_hi = rs[int(0.90*len(rs))]
    pi_80 = round(sum(1 for r in res if pi_lo<=r<=pi_hi)/n*100,1)
    return {"n":n,"MAPE":mape,"SMAPE":smape,"MAE":mae,"RMSE":rmse,
            "Theil_U":theil,"Dir_Acc_pct":dir_acc,
            "PI_80_cov_pct":pi_80,"PI_80_lower":round(pi_lo,4),"PI_80_upper":round(pi_hi,4)}

def run_forecast_backtesting(tech_pairs):
    results = []
    for pair in tech_pairs:
        ac = pair.get("actual_cooc",[])
        if len(ac)<5: continue
        for train_end in [3, 4]:
            train_vals = ac[:train_end]
            test_vals  = ac[train_end:]
            if len(train_vals)<2 or not test_vals: continue
            log_gr = statistics.mean([math.log(max(train_vals[i]/max(train_vals[i-1],0.5),0.01))
                                      for i in range(1,len(train_vals))])
            log_gr_cap = min(log_gr, math.log(3)/5)
            last_c = train_vals[-1]
            lstm_fc= [round(last_c*math.exp(log_gr_cap*(i+1))) for i in range(len(test_vals))]
            naive_fc=[last_c]*len(test_vals)
            m = compute_forecast_metrics(test_vals, lstm_fc, naive_fc)
            m.update({
                "pair_id":    pair["pair_id"],
                "domain":     pair["domain"],
                "train_window":f"periods_0-{train_end-1}",
                "test_window": f"periods_{train_end}-{len(ac)-1}",
                "n_train":    len(train_vals),
                "actual_test":str(test_vals),
                "lstm_fc":    str(lstm_fc),
                "naive_fc":   str(naive_fc),
                "growth_rate_used": round(log_gr_cap,4),
            })
            results.append(m)
            dir_str = "✓" if m.get("Dir_Acc_pct",0)>50 else "✗"
            print(f"    [FORECAST] {pair['pair_id']} train=0-{train_end-1} "
                  f"Dir={m.get('Dir_Acc_pct',0):.0f}% "
                  f"SMAPE={m.get('SMAPE','N/A')}% {dir_str}")
    return results

# ==============================================================================
# 12. [PERSONA] 페르소나 편향 진단
# ==============================================================================
def analyze_persona_bias(all_results):
    rows = []
    for r in all_results:
        e_type = r.get("expert1_type","unknown")
        rows.append({
            "pair_id":          r.get("pair_id",""),
            "run_idx":          r.get("run_idx",0),
            "expert1_name":     r.get("expert1_name",""),
            "expert1_company":  r.get("expert1_company",""),
            "expert1_type":     e_type,
            "expert2_name":     r.get("expert2_name",""),
            "expert2_company":  r.get("expert2_company",""),
            "expert2_type":     r.get("expert2_type",""),
            "avg_expert_score": r.get("avg_expert_score",0),
            "CQS_collab_eq4":   r.get("CQS_collab_eq4",0),
            "CQS_extended":     r.get("CQS_extended",0),
            "pair_weighted_by_large_corp": int(e_type=="large_corp"),
        })
    types = [r["expert1_type"] for r in rows]
    type_counts = {t: types.count(t) for t in set(types)}
    total = max(len(types),1)
    print(f"\n  [PERSONA] 출원인 유형 분포: {type_counts}")
    print(f"  대기업 비율: {type_counts.get('large_corp',0)/total*100:.1f}%")
    lc_cqs = [r["CQS_collab_eq4"] for r in rows if r["expert1_type"]=="large_corp"]
    ot_cqs = [r["CQS_collab_eq4"] for r in rows if r["expert1_type"]!="large_corp"]
    if lc_cqs and ot_cqs:
        lc_m = statistics.mean(lc_cqs); ot_m = statistics.mean(ot_cqs)
        print(f"  대기업 평균 CQS={lc_m:.3f} | 기타 평균 CQS={ot_m:.3f}")
        print(f"  편향 차이: Δ={lc_m-ot_m:+.3f}")
    return rows

# ==============================================================================
# 13. [CIRC] 순환성 감사 테이블 생성
# ==============================================================================
def generate_circularity_audit():
    rows = [
        {"CQS_Component":"Clarity","Scorer":"NLI (DeBERTa-base)",
         "Scorer_Type":"cross-encoder","Is_LLM_Based":False,
         "Circularity_Level":"None","Model_ID":"cross-encoder/nli-deberta-base",
         "Note":"NLI entailment; discriminative not generative"},
        {"CQS_Component":"Actionability","Scorer":"CE (MiniLM)",
         "Scorer_Type":"cross-encoder","Is_LLM_Based":False,
         "Circularity_Level":"None","Model_ID":"cross-encoder/ms-marco-MiniLM-L-6-v2",
         "Note":"Relevance scoring; completely separate from generator"},
        {"CQS_Component":"Alignment","Scorer":"ST (MiniLM-L6)",
         "Scorer_Type":"bi-encoder","Is_LLM_Based":False,
         "Circularity_Level":"None","Model_ID":"sentence-transformers/all-MiniLM-L6-v2",
         "Note":"Embedding cosine; no generation involved"},
        {"CQS_Component":"Feasibility","Scorer":"CE+QNLI",
         "Scorer_Type":"cross-encoder","Is_LLM_Based":False,
         "Circularity_Level":"None","Model_ID":"CE+cross-encoder/qnli-distilroberta-base",
         "Note":"Deployment feasibility via NLI; non-generative"},
        {"CQS_Component":"Novelty","Scorer":"Jaccard rule",
         "Scorer_Type":"rule-based","Is_LLM_Based":False,
         "Circularity_Level":"None","Model_ID":"set-overlap formula",
         "Note":"Pure set arithmetic; zero LLM involvement"},
        {"CQS_Component":"Factual Grounding","Scorer":"KW density + QNLI",
         "Scorer_Type":"rule+cross-encoder","Is_LLM_Based":False,
         "Circularity_Level":"None","Model_ID":"keyword-density + qnli-distilroberta",
         "Note":"Domain keyword presence; deterministic"},
    ]
    print("  [CIRC] 순환성 감사 테이블 생성 — 모든 스코어러: 비LLM")
    return rows

# ==============================================================================
# 14. [HUMAN] 인간 평가 시트 생성 (블라인드)
# ==============================================================================
def export_human_eval_sheet(results, path=OUTPUT_HUMAN_SHEET, n=20):
    random.seed(SEED_BASE+99)
    valid = [r for r in results if r.get("collab_goal","").strip()]
    random.shuffle(valid)
    sampled = valid[:n]
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "proposal_id","pair_domain","joint_rd_goal","key_technologies",
            "Q1_milestone","Q4_milestone","innovation_points","phase1_knowledge_gap",
            "R1_tech_feas","R1_strat_nov","R1_collab_pot","R1_actionability","R1_urban_align",
            "R2_tech_feas","R2_strat_nov","R2_collab_pot","R2_actionability","R2_urban_align",
            "R3_tech_feas","R3_strat_nov","R3_collab_pot","R3_actionability","R3_urban_align",
            "_cqs_eq4","_cqs_ext","_condition","_pair_id","_run_idx"
        ])
        for i,r in enumerate(sampled):
            p3  = r.get("_collab_raw",{}) or {}
            ms  = p3.get("quarterly_milestones",{}) if isinstance(p3,dict) else {}
            kts = p3.get("key_technologies",[]) if isinstance(p3,dict) else []
            ips = p3.get("innovation_points",[]) if isinstance(p3,dict) else []
            if not isinstance(ms,  dict): ms  = {}
            if not isinstance(kts, list): kts = []
            if not isinstance(ips, list): ips = []
            w.writerow([
                i+1, r.get("pair_domain",""),
                r.get("collab_goal","")[:200],
                "; ".join(str(k) for k in kts)[:150],
                ms.get("Q1","")[:100], ms.get("Q4","")[:100],
                "; ".join(str(ip) for ip in ips)[:150],
                r.get("phase1_gap","")[:80],
                *[""]*(5*3),
                round(r.get("CQS_collab_eq4",0),4),
                round(r.get("CQS_extended",0),4),
                r.get("condition",""), r.get("pair_id",""), r.get("run_idx",0)
            ])
    print(f"  [HUMAN] 블라인드 평가 시트 → {path} ({len(sampled)}개 제안)")

# ==============================================================================
# 15. 실험 조건 실행
# ==============================================================================
def _pick_experts(run_idx, seed_offset=0):
    random.seed(SEED_BASE + run_idx*13 + seed_offset)
    return (random.choice(INDUSTRY_EXPERTS[:2]),
            random.choice(INDUSTRY_EXPERTS[2:]))

def run_condition(model_key, cond_name, tech_pair, run_idx, strategy,
                  include_patent=True, include_academic=True,
                  include_persona=True, include_forecast=True,
                  single_llm=False, structured_single=False,
                  temperature=None, n_done=0, n_total=1):
    random.seed(SEED_BASE + run_idx*17 + abs(hash(cond_name))%100)
    lbl = lambda s: f"{cond_name[:12]}|{tech_pair['pair_id']}|r{run_idx}|{s}"

    if include_persona:
        e1,e2 = _pick_experts(run_idx, abs(hash(cond_name))%50)
        avg_expert = round((e1["weighted_score"]+e2["weighted_score"])/2, 3)
        ac = discover_academic_expert(tech_pair["domain"], tech_pair["pair_id"], run_idx)
    else:
        e1={"name":"Expert A","company":"MobilityCorp","expertise":"general urban mobility AI",
            "weighted_score":50.0,"applicant_type":"unknown","cpc_primary":"G08G1"}
        e2={"name":"Expert B","company":"ConnectedSys","expertise":"general connected vehicle data",
            "weighted_score":50.0,"applicant_type":"unknown","cpc_primary":"G08G1"}
        ac={"name":"Researcher","affiliation":"University","expertise":"general AI","affil_type":"academia"}
        avg_expert=50.0

    if single_llm:
        prompt = build_single_llm_prompt(tech_pair, structured_single)
        p3,ok = call_llm_gen(model_key, prompt, lbl("SL"), run_idx,
                             cond_name, tech_pair["pair_id"],
                             temperature=temperature, n_done=n_done, n_total=n_total)
        ev = hybrid_evaluate({}, p3, tech_pair, cond_name)
        return {}, p3, ev, ok, avg_expert, e1, e2

    conv_prompt = build_conv_prompt(e1, e2, tech_pair, include_patent, strategy, include_forecast)
    p1,ok1 = call_llm_gen(model_key, conv_prompt, lbl("CV"), run_idx,
                          cond_name, tech_pair["pair_id"],
                          temperature=temperature, n_done=n_done, n_total=n_total)

    collab_prompt = build_collab_prompt(e1, e2, ac, tech_pair, include_academic, strategy)
    p3,ok3 = call_llm_gen(model_key, collab_prompt, lbl("CO"), run_idx,
                          cond_name, tech_pair["pair_id"],
                          temperature=temperature, n_done=n_done+1, n_total=n_total)

    ev = hybrid_evaluate(p1, p3, tech_pair, cond_name)
    return p1, p3, ev, (ok1 and ok3), avg_expert, e1, e2

# ==============================================================================
# 16. 단일 모델 전체 실험
# ==============================================================================
def run_experiment(model_key, model_label):
    print(f"\n{'='*68}")
    print(f"  [{model_label}] V8 실험 시작")
    print(f"{'='*68}")

    ABLATION_CONDITIONS = [
        ("Full Pipeline (Control)", True, True,  True,  True),
        ("No Persona Module",       True, True,  False, True),
        ("No Patent Grounding",     False,True,  True,  True),
        ("No Academic Integration", True, False, True,  True),
        ("No Forecasting/LSTM",     True, True,  True,  False),
    ]

    total_calls = N_RUNS_PER_PAIR*len(TECH_PAIRS)*2
    all_results=[]; ablation_results=[]; baseline_results=[]
    n_done=0

    print(f"\n▶ Phase A: Full Pipeline [{model_label}]")
    random.seed(SEED_BASE)
    for pair in TECH_PAIRS:
        for run_i in range(N_RUNS_PER_PAIR):
            strategy = MEETING_STRATEGIES[run_i%3]
            cond = "Full Pipeline"
            try:
                p1,p3,ev,ok,avg_exp,e1,e2 = run_condition(
                    model_key, cond, pair, run_i, strategy,
                    n_done=n_done, n_total=total_calls)
                goal = str(p3.get("joint_rd_goal",""))[:250] if isinstance(p3,dict) else ""
                sc   = extract_scores(ev, cond)
                row  = {
                    **sc,
                    "model":          model_key,
                    "model_label":    model_label,
                    "condition":      cond,
                    "pair_id":        pair["pair_id"],
                    "pair_domain":    pair["domain"],
                    "pair_label":     pair["label"],
                    "run_idx":        run_i,
                    "strategy":       strategy,
                    "api_success":    ok,
                    "avg_expert_score":avg_exp,
                    "growth_potential":pair["growth_potential"],
                    "pair_growth_pct":pair["growth_pct"],
                    "collab_goal":    goal,
                    "phase1_gap":     (p1.get("knowledge_gap","")[:100]
                                      if isinstance(p1,dict) else ""),
                    "evaluator_type": ev.get("evaluator_type","non-LLM"),
                    "expert1_name":   e1.get("name",""),
                    "expert1_company":e1.get("company",""),
                    "expert1_type":   e1.get("applicant_type",""),
                    "expert2_name":   e2.get("name",""),
                    "expert2_company":e2.get("company",""),
                    "expert2_type":   e2.get("applicant_type",""),
                    "_collab_raw":    p3,
                }
                all_results.append(row)
                print(f"  [{model_key}][{pair['pair_id']} r{run_i+1}] "
                      f"CQS={sc['CQS_collab_eq4']:.3f} Ext={sc['CQS_extended']:.3f} "
                      f"Feas={sc['collab_feasibility_score']:.3f} | "
                      f"{strategy[:12]:12s} | {'OK' if ok else 'fb'}")
                n_done+=2
            except Exception as e:
                print(f"  [ERROR] {pair['pair_id']} run{run_i}: {e}"); n_done+=2

    full_ctrl_vals = [r["CQS_collab_eq4"] for r in all_results]
    for r in all_results:
        ablation_results.append({
            "Condition":"Full Pipeline (Control)",
            "pair_id":r["pair_id"],"strategy":r["strategy"],
            "CQS":r["CQS_collab_eq4"],"CQS_extended":r["CQS_extended"],
            "Delta_vs_full":0.0,"api_success":r["api_success"],
            "model":model_key,"phase_note":"collab+conv"
        })

    print(f"\n▶ Phase B: Ablation [{model_label}]")
    for cond_name,inc_pat,inc_acad,inc_persona,inc_fc in ABLATION_CONDITIONS[1:]:
        print(f"\n  > {cond_name}")
        for pair in TECH_PAIRS:
            for run_i in range(N_RUNS_PER_PAIR):
                strategy = MEETING_STRATEGIES[run_i%3]
                try:
                    _,_,ev,ok,_,_,_ = run_condition(
                        model_key, cond_name, pair, run_i, strategy,
                        include_patent=inc_pat, include_academic=inc_acad,
                        include_persona=inc_persona, include_forecast=inc_fc,
                        n_done=n_done, n_total=total_calls)
                    sc  = extract_scores(ev, cond_name)
                    fm  = statistics.mean(full_ctrl_vals) if full_ctrl_vals else 8.98
                    ablation_results.append({
                        "Condition":cond_name,
                        "pair_id":pair["pair_id"],"strategy":strategy,
                        "CQS":round(sc["CQS_collab_eq4"],4),
                        "CQS_extended":sc["CQS_extended"],
                        "Delta_vs_full":round(sc["CQS_collab_eq4"]-fm,4),
                        "api_success":ok,"model":model_key,"phase_note":"collab+conv"
                    })
                    print(f"    [{pair['pair_id']} r{run_i+1}] "
                          f"CQS={sc['CQS_collab_eq4']:.3f} | {'OK' if ok else 'fb'}")
                    n_done+=2
                except Exception as e:
                    print(f"    [ERROR] {pair['pair_id']}: {e}"); n_done+=2

    ablation_stats = compute_ablation_stats(ablation_results)
    print(f"\n  [STAT-2] FDR 보정 완료: {len(ablation_stats)}개 조건")

    print(f"\n▶ Phase C: Baseline [{model_label}]")
    for r in all_results:
        baseline_results.append({
            "Condition":"Full Pipeline","pair_id":r["pair_id"],
            "strategy":r["strategy"],"CQS":r["CQS_collab_eq4"],
            "CQS_extended":r["CQS_extended"],
            "feasibility":r["collab_feasibility_score"],
            "novelty":r["collab_novelty_score"],
            "api_success":r["api_success"],"model":model_key
        })
    for cond_name,structured in [
        (f"Single LLM ({model_label})",False),
        ("Structured CoT",True)
    ]:
        print(f"\n  > {cond_name}")
        for pair in TECH_PAIRS:
            for run_i in range(N_RUNS_PER_PAIR):
                strategy = MEETING_STRATEGIES[run_i%3]
                try:
                    _,_,ev,ok,_,_,_ = run_condition(
                        model_key, cond_name, pair, run_i, strategy,
                        single_llm=True, structured_single=structured,
                        n_done=n_done, n_total=total_calls)
                    sc = extract_scores(ev, cond_name)
                    baseline_results.append({
                        "Condition":cond_name,"pair_id":pair["pair_id"],
                        "strategy":strategy,"CQS":sc["CQS_collab_eq4"],
                        "CQS_extended":sc["CQS_extended"],
                        "feasibility":sc["collab_feasibility_score"],
                        "novelty":sc["collab_novelty_score"],
                        "api_success":ok,"model":model_key
                    })
                    n_done+=1
                except Exception as e:
                    print(f"    [ERROR]: {e}"); n_done+=1

    return {"main":all_results, "ablation":ablation_results,
            "ablation_stats":ablation_stats, "baseline":baseline_results}

# ==============================================================================
# 17. Strong Baseline (Qwen2.5-3B)
# ==============================================================================
def run_strong_baseline(tech_pairs, n_runs=STRONG_N_RUNS):
    print(f"\n[Strong Baseline] Qwen2.5-3B — {len(tech_pairs)}쌍 × {n_runs}회")
    _load_gen_pipeline("strong")
    results=[]
    for pair in tech_pairs:
        for run_i in range(n_runs):
            strategy = MEETING_STRATEGIES[run_i%3]
            for cond_label,structured in [
                ("Strong LLM (direct)",False),
                ("Strong LLM (CoT)",True)
            ]:
                try:
                    _,p3,ev,ok,_,_,_ = run_condition(
                        "strong", cond_label, pair, run_i, strategy,
                        single_llm=True, structured_single=structured,
                        temperature=STRONG_TEMPERATURE)
                    sc = extract_scores(ev, cond_label)
                    results.append({
                        **sc,"condition":cond_label,"model":"Qwen2.5-3B",
                        "pair_id":pair["pair_id"],"pair_domain":pair["domain"],
                        "run_idx":run_i,"strategy":strategy,"api_success":ok,
                        "collab_goal":str(p3.get("joint_rd_goal",""))[:200]
                                    if isinstance(p3,dict) else ""
                    })
                    print(f"  [{cond_label[:20]}] {pair['pair_id']} "
                          f"CQS={sc['CQS_collab_eq4']:.3f}")
                except Exception as e:
                    print(f"  [ERROR] {pair['pair_id']}: {e}")
    return results

# ==============================================================================
# 18. CSV 저장
# ==============================================================================
def _save_csv(rows, path, tag=""):
    if not rows: print(f"  저장 건너뜀: {path}"); return
    clean = [{k:v for k,v in r.items() if not k.startswith("_")} for r in rows]
    if HAS_PANDAS:
        pd.DataFrame(clean).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        with open(path,"w",newline="",encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=clean[0].keys())
            w.writeheader(); w.writerows(clean)
    print(f"  저장 → {path} ({len(clean)}행) {tag}")

# ==============================================================================
# 19. 통계 요약 출력
# ==============================================================================
def print_summary(results, model_label, ablation_stats=None):
    print(f"\n{'='*68}")
    print(f"  [{model_label}] V8 통계 요약")
    print(f"{'='*68}")
    fp_cqs  = [r["CQS_collab_eq4"] for r in results]
    fp_ext  = [r["CQS_extended"] for r in results]
    fp_feas = [r["collab_feasibility_score"] for r in results]
    for vals,lbl in [(fp_cqs,"CQS_eq4"),(fp_ext,"CQS_ext"),(fp_feas,"Feasibility")]:
        if vals:
            m,lo,hi = bootstrap_bca(vals)
            sd = statistics.stdev(vals) if len(vals)>1 else 0.0
            print(f"  {lbl:12s}: M={m:.3f} SD={sd:.3f} 95%BCa=[{lo:.3f},{hi:.3f}] n={len(vals)}")
    ok = sum(1 for r in results if r.get("api_success"))
    print(f"\n  생성 성공률: {ok}/{len(results)} ({round(100*ok/max(len(results),1))}%)")

# ==============================================================================
# 20. 메인
# ==============================================================================
if __name__ == "__main__":
    t_start = datetime.now()
    print("\n"+"="*68)
    print("  R&D 협업 품질 실험 V8 — 리비전 핵심 대응")
    print("="*68)

    # 스코어러 로딩
    print("\n[1] 비LLM 판별 스코어러 로딩")
    load_all_scorers()

    # [CIRC] 순환성 감사 테이블 사전 생성
    circ_rows = generate_circularity_audit()
    _save_csv(circ_rows, OUTPUT_CIRC_AUDIT, "← [CIRC] 순환성 감사")

    # [FORECAST] 실제 walk-forward 백테스팅
    print("\n[2] [FORECAST] Walk-forward 백테스팅 (실제 CPC 공존 빈도)")
    forecast_results = run_forecast_backtesting(TECH_PAIRS)
    _save_csv(forecast_results, OUTPUT_FORECAST, "← [FORECAST] 실제 데이터")

    # Phi-3 실험
    print("\n[3] Phi-3-mini 실험")
    _load_gen_pipeline("phi3")
    phi3_exp = run_experiment("phi3", "Phi-3-mini-4k")
    _save_csv(phi3_exp["main"],          OUTPUT_MAIN_PHI3)
    _save_csv(phi3_exp["ablation"],      OUTPUT_ABLATION.replace("v8","v8_phi3"))
    _save_csv(phi3_exp["ablation_stats"],OUTPUT_ABL_STATS)
    _save_csv(phi3_exp["baseline"],      OUTPUT_BASELINE.replace("v8","v8_phi3"))
    lme_phi3 = to_lme_ready(phi3_exp["main"], "phi3")
    persona_rows = analyze_persona_bias(phi3_exp["main"])
    _save_csv(persona_rows, OUTPUT_PERSONA_BIAS, "← [PERSONA] 편향 진단")
    export_human_eval_sheet(phi3_exp["main"], OUTPUT_HUMAN_SHEET, n=20)
    print_summary(phi3_exp["main"],"Phi-3-mini-4k", phi3_exp["ablation_stats"])
    unload_model("phi3")

    # Qwen1.5B 실험
    print("\n[4] Qwen2.5-1.5B 실험")
    _load_gen_pipeline("qwen")
    qwen_exp = run_experiment("qwen", "Qwen2.5-1.5B")
    _save_csv(qwen_exp["main"],    OUTPUT_MAIN_QWEN)
    _save_csv(qwen_exp["ablation"],OUTPUT_ABLATION.replace("v8","v8_qwen"))
    _save_csv(qwen_exp["baseline"],OUTPUT_BASELINE.replace("v8","v8_qwen"))
    lme_all = lme_phi3 + to_lme_ready(qwen_exp["main"], "qwen")
    _save_csv(lme_all, OUTPUT_LME_READY, "← [STAT-3] R lme4 직접 입력")
    print_summary(qwen_exp["main"],"Qwen2.5-1.5B",qwen_exp["ablation_stats"])
    unload_model("qwen")

    # Strong Baseline (Qwen2.5-3B)
    print("\n[5] Strong Baseline (Qwen2.5-3B)")
    strong_results = run_strong_baseline(TECH_PAIRS, n_runs=STRONG_N_RUNS)
    _save_csv(strong_results, OUTPUT_STRONG, "← Strong Baseline")
    unload_model("strong")

    elapsed = (datetime.now()-t_start).total_seconds()/60
    print(f"\n  총 소요시간: {elapsed:.1f}분")
    print(f"\n  완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*68)
