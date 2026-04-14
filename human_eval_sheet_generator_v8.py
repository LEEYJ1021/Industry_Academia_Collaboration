"""
human_eval_sheet_v8.csv 완전 재생성 — 3개 로컬 모델 통합
==========================================================
- results_main_v8_phi3.csv  → Phi-3-mini    로 누락 필드 생성
- results_main_v8_qwen.csv  → Qwen2.5-1.5B 로 누락 필드 생성
- results_strong_v8.csv     → Qwen2.5-3B   로 누락 필드 생성
- 전체 병합 → human_eval_sheet_v8.csv 하나로 저장
- Anthropic API 불필요, 추가 설치 불필요
"""

import os, gc, json, re, csv, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==============================================================================
# 경로 / 설정
# ==============================================================================
BASE_DIR = r"C:\Users\USER\Desktop\논문\후속논문(10) [1차연구] LangGraph_Use-Cases-Research-Assistant"

# 모델별 입력 CSV
MODEL_CSV_MAP = [
    ("phi3",   "microsoft/Phi-3-mini-4k-instruct",  "results_main_v8_phi3.csv"),
    ("qwen",   "Qwen/Qwen2.5-1.5B-Instruct",        "results_main_v8_qwen.csv"),
    ("strong", "Qwen/Qwen2.5-3B-Instruct",           "results_strong_v8.csv"),
]

OUTPUT_CSV  = os.path.join(BASE_DIR, "human_eval_sheet_v8.csv")
SEED_BASE   = 42
N_SAMPLE    = 20      # 모델당 샘플 수 (전체 = N_SAMPLE × 모델 수)
MAX_TOKENS  = 300
TEMPERATURE = 0.2

# ==============================================================================
# 기술 쌍 메타데이터
# ==============================================================================
TECH_PAIRS_META = {
    "G06F16-H04W12": {
        "label":    "User Profile-Based Intelligent Traffic Services",
        "cpc_a_desc": "Database Technologies / Data Retrieval",
        "cpc_b_desc": "Wireless Network Security",
    },
    "B60R21-B60R23": {
        "label":    "Advanced Visual Safety and Parking Assistance Systems",
        "cpc_a_desc": "Occupant Safety / Air-bags",
        "cpc_b_desc": "Vehicle Viewing Arrangements / Cameras",
    },
    "G08G5-H04L20": {
        "label":    "Broadcast-Based Air Traffic Control and UAV Systems",
        "cpc_a_desc": "Air Traffic Control Systems",
        "cpc_b_desc": "Broadcast Data Transmission Protocols",
    },
    "B60R23-G01S7": {
        "label":    "Sensor Fusion for Robust Obstacle Detection",
        "cpc_a_desc": "Vehicle Viewing Arrangements",
        "cpc_b_desc": "Radar/LiDAR Detection Systems",
    },
    "B61L25-G06Q50": {
        "label":    "Intelligent Railway Operation",
        "cpc_a_desc": "Railway Control Systems",
        "cpc_b_desc": "Service Management Systems",
    },
    "G07C5-H04M1": {
        "label":    "Telematics for Vehicle Fleet Management",
        "cpc_a_desc": "Vehicle Data Recording",
        "cpc_b_desc": "Telephonic Communication",
    },
}

FIELDNAMES = [
    "proposal_id", "source_model", "pair_domain",
    "joint_rd_goal", "key_technologies",
    "Q1_milestone", "Q4_milestone", "innovation_points",
    "phase1_knowledge_gap",
    "R1_tech_feas", "R1_strat_nov", "R1_collab_pot",
    "R1_actionability", "R1_urban_align",
    "R2_tech_feas", "R2_strat_nov", "R2_collab_pot",
    "R2_actionability", "R2_urban_align",
    "R3_tech_feas", "R3_strat_nov", "R3_collab_pot",
    "R3_actionability", "R3_urban_align",
    "_cqs_eq4", "_cqs_ext", "_condition", "_pair_id", "_run_idx",
]

# ==============================================================================
# 모델 로딩 / 언로드
# ==============================================================================
def load_pipeline(model_id: str, model_key: str):
    print(f"\n  [로딩] {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="cpu",
        trust_remote_code=True, attn_implementation="eager",
    )
    mdl.eval()
    pipe = pipeline("text-generation", model=mdl,
                    tokenizer=tok, return_full_text=False)
    print(f"  [로딩 완료] ✓ {model_id}")
    return pipe

def unload_pipeline(pipe):
    try:
        del pipe.model
    except Exception:
        pass
    del pipe
    gc.collect()
    print("  [언로드] 완료")

# ==============================================================================
# 프롬프트 / 파싱
# ==============================================================================
_SYS = ("You are a precise JSON-only assistant. "
        "Always respond with a single valid JSON object. "
        "No explanation, no markdown, no text outside the JSON.")

def make_prompt(model_key: str, goal: str, gap: str, meta: dict) -> str:
    user = (
        f"Given the R&D goal below, extract structured details.\n\n"
        f"Goal: {goal}\n"
        f"Knowledge Gap: {gap}\n"
        f"Tech A: {meta.get('cpc_a_desc','')}\n"
        f"Tech B: {meta.get('cpc_b_desc','')}\n"
        f"Focus: {meta.get('label','')}\n\n"
        f"Return ONLY valid JSON:\n"
        f'{{"key_technologies":["tech1","tech2","tech3"],'
        f'"Q1_milestone":"concrete Q1 deliverable (<80 chars)",'
        f'"Q4_milestone":"final deliverable: paper or patent (<80 chars)",'
        f'"innovation_points":["novelty1","novelty2"]}}'
    )
    if model_key == "phi3":
        return (f"<|system|>\n{_SYS}\n<|end|>\n"
                f"<|user|>\n{user}\n<|end|>\n<|assistant|>\n")
    else:  # qwen / strong (같은 포맷)
        return (f"<|im_start|>system\n{_SYS}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n")

def clean_raw(raw: str, model_key: str) -> str:
    tags = {"phi3":  ["<|assistant|>","<|end|>","<|endoftext|>"],
            "qwen":  ["<|im_end|>","<|endoftext|>"],
            "strong":["<|im_end|>","<|endoftext|>"]}
    for t in tags.get(model_key, []):
        if t in raw:
            raw = raw.split(t)[0].strip()
    return raw

def parse_json(text: str) -> dict:
    for fn in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.sub(r"```(?:json)?\s*|\s*```", "", t).strip()),
        lambda t: json.loads(re.search(r"\{.*\}", t, re.DOTALL).group()),
    ]:
        try:
            return fn(text)
        except Exception:
            pass

    # 정규식 폴백
    result = {}
    m = re.search(r'"key_technologies"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
    if m:
        result["key_technologies"] = re.findall(r'"([^"]+)"', m.group(1))
    for k in ["Q1_milestone", "Q4_milestone"]:
        m = re.search(rf'"{k}"\s*:\s*"([^"]*)"', text)
        if m:
            result[k] = m.group(1)
    m = re.search(r'"innovation_points"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
    if m:
        result["innovation_points"] = re.findall(r'"([^"]+)"', m.group(1))
    return result

def rule_fallback(goal: str, meta: dict) -> dict:
    """추론 완전 실패 시 키워드 기반 폴백."""
    cpc_a = meta.get("cpc_a_desc", "Technology A").split("/")[0].strip()
    cpc_b = meta.get("cpc_b_desc", "Technology B").split("/")[0].strip()
    label = meta.get("label", "")
    gl    = goal.lower()
    kw_map = {
        "AI/ML integration":      ["ai","machine learning","neural","deep"],
        "sensor fusion":          ["sensor","fusion","lidar","radar","camera"],
        "edge computing":         ["edge","cloud","distributed","real-time"],
        "V2X communication":      ["v2x","vehicle","network","wireless"],
        "UAV navigation":         ["uav","drone","air","autonomous"],
        "federated learning":     ["federated","privacy","secure"],
        "foundation model":       ["foundation","llm","large model","language"],
        "real-time optimization": ["optim","real-time","adaptive","efficient"],
    }
    techs = [t for t, kws in kw_map.items() if any(k in gl for k in kws)][:3]
    while len(techs) < 3:
        techs.append(f"{cpc_a} integration" if len(techs)==0 else
                     f"{cpc_b} processing"  if len(techs)==1 else
                     "data-driven decision support")
    return {
        "key_technologies": techs,
        "Q1_milestone": f"Establish baseline dataset and architecture for {label[:40]}",
        "Q4_milestone": "Submit joint patent and publish conference paper",
        "innovation_points": [
            f"Cross-domain convergence of {cpc_a} and {cpc_b}",
            "Adaptive AI-driven framework for real-time optimization",
        ],
    }

# ==============================================================================
# 단일 행 추론
# ==============================================================================
def infer_row(pipe, model_key: str, goal: str, gap: str, meta: dict) -> dict:
    prompt = make_prompt(model_key, goal[:300], gap[:100], meta)
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = pipe(
                    prompt, max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE, do_sample=(TEMPERATURE > 0.05),
                    top_p=0.9, repetition_penalty=1.05,
                    pad_token_id=pipe.tokenizer.eos_token_id,
                    eos_token_id=pipe.tokenizer.eos_token_id,
                    use_cache=False,
                )
            raw    = clean_raw(str(out[0]["generated_text"]).strip(), model_key)
            parsed = parse_json(raw)
            if parsed and any(k in parsed for k in
                              ["key_technologies","Q1_milestone","Q4_milestone"]):
                print(f"        ✓ ({time.perf_counter()-t0:.1f}s, 시도 {attempt+1})")
                return parsed
            print(f"        ✗ JSON 불완전 (시도 {attempt+1})")
        except Exception as e:
            print(f"        ✗ 오류 (시도 {attempt+1}): {str(e)[:50]}")
    print("        → 규칙 폴백")
    return rule_fallback(goal, meta)

# ==============================================================================
# CSV 읽기 (phi3/qwen main 형식 + strong 형식 통합)
# ==============================================================================
def load_csv(path: str) -> list:
    if not os.path.exists(path):
        print(f"  [건너뜀] 파일 없음: {path}")
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # strong CSV는 collab_goal 대신 다른 컬럼명일 수 있음 → 통일
    for r in rows:
        if "collab_goal" not in r and "joint_rd_goal" in r:
            r["collab_goal"] = r["joint_rd_goal"]
        if "phase1_gap" not in r:
            r["phase1_gap"] = ""
        # condition 컬럼 통일 (strong은 'condition' 소문자)
        if "condition" not in r and "Condition" in r:
            r["condition"] = r["Condition"]
    return rows

def safe_join(val, sep="; ", maxlen=150) -> str:
    if isinstance(val, list):
        return sep.join(str(v).strip() for v in val if v)[:maxlen]
    if isinstance(val, dict):
        return sep.join(f"{k}: {v}" for k, v in val.items())[:maxlen]
    return str(val or "")[:maxlen]

def safe_float(v, default=0.0) -> float:
    try:
        return round(float(v), 4)
    except (ValueError, TypeError):
        return default

# ==============================================================================
# 모델별 처리 → 결과 리스트 반환
# ==============================================================================
def process_model(model_key: str, model_id: str, csv_path: str) -> list:
    full_path = os.path.join(BASE_DIR, csv_path)
    rows = load_csv(full_path)
    if not rows:
        return []

    # collab_goal 있는 행만 필터링 + 셔플 + 샘플링
    valid = [r for r in rows if r.get("collab_goal", "").strip()]
    print(f"  유효 행: {len(valid)}개")
    random.seed(SEED_BASE + abs(hash(model_key)) % 100)
    random.shuffle(valid)
    sampled = valid[:N_SAMPLE]

    # 모델 로딩
    pipe = load_pipeline(model_id, model_key)

    results = []
    for i, r in enumerate(sampled):
        pair_id    = r.get("pair_id", "")
        collab_goal = r.get("collab_goal", "").strip()
        phase1_gap  = r.get("phase1_gap", "").strip()
        meta        = TECH_PAIRS_META.get(pair_id, {})

        print(f"    [{i+1:2d}/{len(sampled)}] {pair_id} | {meta.get('label','')[:35]}")
        structured = infer_row(pipe, model_key, collab_goal, phase1_gap, meta)

        results.append({
            "source_model":         model_key,
            "pair_domain":          r.get("pair_domain", ""),
            "joint_rd_goal":        collab_goal[:200],
            "key_technologies":     safe_join(structured.get("key_technologies", [])),
            "Q1_milestone":         str(structured.get("Q1_milestone", ""))[:100],
            "Q4_milestone":         str(structured.get("Q4_milestone", ""))[:100],
            "innovation_points":    safe_join(structured.get("innovation_points", [])),
            "phase1_knowledge_gap": phase1_gap[:80],
            # CQS / 메타
            "_cqs_eq4":  safe_float(r.get("CQS_collab_eq4", r.get("CQS", 0))),
            "_cqs_ext":  safe_float(r.get("CQS_extended", 0)),
            "_condition": r.get("condition", r.get("Condition", "")),
            "_pair_id":   pair_id,
            "_run_idx":   r.get("run_idx", ""),
        })

    # 모델 언로드 (메모리 확보)
    unload_pipeline(pipe)
    return results

# ==============================================================================
# 메인
# ==============================================================================
def main():
    print("=" * 68)
    print("  human_eval_sheet_v8.csv 재생성 — 3개 로컬 모델 통합")
    print("=" * 68)

    os.makedirs(BASE_DIR, exist_ok=True)
    all_rows = []

    # ── 모델별 순차 처리 (메모리 절약을 위해 하나씩 로드/언로드) ──────────
    for model_key, model_id, csv_file in MODEL_CSV_MAP:
        print(f"\n{'─'*68}")
        print(f"  모델: {model_key.upper()} | CSV: {csv_file}")
        print(f"{'─'*68}")
        rows = process_model(model_key, model_id, csv_file)
        print(f"  → {len(rows)}행 생성 완료")
        all_rows.extend(rows)

    if not all_rows:
        print("\n[오류] 생성된 행이 없습니다. CSV 파일 경로를 확인하세요.")
        return

    # ── 블라인드를 위한 전체 셔플 + proposal_id 재부여 ───────────────────
    random.seed(SEED_BASE + 999)
    random.shuffle(all_rows)

    print(f"\n{'─'*68}")
    print(f"  전체 {len(all_rows)}행 → {OUTPUT_CSV} 저장 중...")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for idx, row in enumerate(all_rows, start=1):
            writer.writerow({
                "proposal_id": idx,
                **{k: row.get(k, "") for k in FIELDNAMES if k != "proposal_id"},
                # 평가자 입력란 (빈칸)
                "R1_tech_feas": "", "R1_strat_nov": "", "R1_collab_pot": "",
                "R1_actionability": "", "R1_urban_align": "",
                "R2_tech_feas": "", "R2_strat_nov": "", "R2_collab_pot": "",
                "R2_actionability": "", "R2_urban_align": "",
                "R3_tech_feas": "", "R3_strat_nov": "", "R3_collab_pot": "",
                "R3_actionability": "", "R3_urban_align": "",
            })

    # ── 완료 요약 ─────────────────────────────────────────────────────────
    from collections import Counter
    model_counts = Counter(r.get("source_model", "") for r in all_rows)
    print(f"\n{'='*68}")
    print(f"  [완료] → {OUTPUT_CSV}")
    print(f"  총 {len(all_rows)}행 | 모든 컬럼 채움")
    print(f"  모델별 행 수:")
    for mk, cnt in model_counts.items():
        print(f"    {mk:8s}: {cnt}행")
    print(f"\n  ※ 평가자 배포 전 _cqs_eq4, _cqs_ext, _condition,")
    print(f"     _pair_id, _run_idx, source_model 컬럼을 숨기세요.")
    print("=" * 68)


if __name__ == "__main__":
    main()
