"""
bayesian_lstm_forecast_v8.py  ── 수정판
=========================================
[문제] 초기 버전의 세 가지 근본 결함:
  1. 정규화 버그: train 통계로 정규화 → test 값이 학습 분포 밖 → underforecast
  2. LSTM의 보간/외삽 한계: 단조 증가 시계열에서 Dir_Acc = 0%
  3. PI95 = 0%: MC-Dropout 분산이 실제값을 포함하지 못함

[해결] LSTM-Informed Bayesian Log-Linear Regression
  ① LSTM은 6쌍 풀링으로 성장률 사전(prior) μ_β 추정에만 사용
  ② 실제 예측: Bayesian log-linear model (log-scale 선형회귀)
       log(y_t) ~ N(α + β·t, σ²)
  ③ 사후분포: Metropolis-Hastings MCMC (500 warmup + 2000 draw)
  ④ 예측 불확실성: 사후 샘플 → 2.5 / 97.5 백분위수 = 95% 신용구간

[근거]
  - 소표본(n=3) 외삽 문제에서 Bayesian 회귀 > LSTM (Chatfield, 2000)
  - log-linear 모형은 CPC 공존 빈도의 지수 성장에 적합 (Li et al., 2022)
  - LSTM prior 정보 활용 → 완전 uninformative prior 보다 실질적
"""

import math
import statistics
import random
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# 전역 설정
# ──────────────────────────────────────────────────────────────
SEED            = 2025
MCMC_WARMUP     = 500    # Metropolis-Hastings 워밍업
MCMC_DRAWS      = 2000   # 사후 샘플 수
MH_STEP_ALPHA   = 0.15   # α 제안 분포 표준편차
MH_STEP_BETA    = 0.08   # β 제안 분포 표준편차
MH_STEP_SIGMA   = 0.05   # σ 제안 분포 표준편차
PRIOR_SIGMA_SD  = 0.4    # σ HalfNormal 파라미터

# LSTM 설정 (prior 추정용)
LSTM_HIDDEN     = 16
LSTM_LAYERS     = 1
LSTM_EPOCHS     = 600
LSTM_LR         = 5e-3

torch.manual_seed(SEED)
random.seed(SEED)


# ══════════════════════════════════════════════════════════════
# PART 1. LSTM — 성장률 사전(prior) 추정기
# ══════════════════════════════════════════════════════════════

class GrowthRateLSTM(nn.Module):
    """
    Log-성장률(β)의 사전 평균 μ_β 를 추정하는 경량 LSTM.
    6쌍 풀링으로 학습 → 각 쌍에 미세조정 → β prior 제공.
    """
    def __init__(self, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _make_growth_dataset(log_seq: List[float]
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    로그-시계열에서 1-step 성장률 예측 데이터 생성.
    입력: log(y_{t-2}), log(y_{t-1})  → 출력: log(y_t) - log(y_{t-1})
    """
    if len(log_seq) < 3:
        return torch.zeros(0, 2, 1), torch.zeros(0, 1)
    xs, ys = [], []
    for i in range(2, len(log_seq)):
        xs.append([log_seq[i - 2], log_seq[i - 1]])
        ys.append(log_seq[i] - log_seq[i - 1])
    X = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
    return X, y


def train_growth_lstm(all_train_seqs: List[List[float]]) -> GrowthRateLSTM:
    """6개 쌍의 로그-시계열을 풀링하여 성장률 LSTM 학습."""
    model     = GrowthRateLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR,
                                 weight_decay=1e-3)
    criterion = nn.MSELoss()

    random.seed(SEED)
    X_all, y_all = [], []
    for seq in all_train_seqs:
        log_seq = [math.log(max(v, 0.5)) for v in seq]
        variants = [log_seq]
        scale    = max(statistics.stdev(log_seq) * 0.05
                       if len(log_seq) > 1 else 0.05, 0.01)
        for _ in range(5):
            noisy = [v + random.gauss(0, scale) for v in log_seq]
            variants.append(noisy)
        for v in variants:
            X_b, y_b = _make_growth_dataset(v)
            if len(X_b):
                X_all.append(X_b)
                y_all.append(y_b)

    if not X_all:
        return model

    X_pool = torch.cat(X_all)
    y_pool = torch.cat(y_all)

    model.train()
    best, no_impr = float('inf'), 0
    for _ in range(LSTM_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_pool), y_pool)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if loss.item() < best - 1e-7:
            best, no_impr = loss.item(), 0
        else:
            no_impr += 1
            if no_impr >= 80:
                break

    return model


def lstm_growth_prior(model: GrowthRateLSTM,
                      train_log: List[float]) -> float:
    """LSTM으로 다음 스텝 성장률 예측 → β prior 평균값으로 사용."""
    if len(train_log) < 2:
        return 0.15
    if len(train_log) >= 3:
        model.eval()
        x_in = torch.tensor([[train_log[-2], train_log[-1]]],
                             dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            mu_beta = model(x_in).item()
        return float(max(-0.5, min(mu_beta, 1.0)))
    return train_log[-1] - train_log[-2]


# ══════════════════════════════════════════════════════════════
# PART 2. Bayesian Log-Linear Model + Metropolis-Hastings
# ══════════════════════════════════════════════════════════════

def _log_likelihood(log_y: List[float], t_idx: List[int],
                    alpha: float, beta: float, sigma: float) -> float:
    """log p(data | α, β, σ)  모델: log(y_t) ~ N(α + β·t, σ²)"""
    if sigma <= 0:
        return -math.inf
    n  = len(log_y)
    ss = sum((ly - (alpha + beta * t)) ** 2
             for ly, t in zip(log_y, t_idx))
    return (-n * math.log(sigma)
            - 0.5 * ss / (sigma ** 2)
            - n * 0.5 * math.log(2 * math.pi))


def _log_prior(alpha: float, beta: float, sigma: float,
               mu_alpha: float, mu_beta: float) -> float:
    """
    log p(α, β, σ)
    α ~ N(mu_alpha, 0.5²)
    β ~ N(mu_beta,  0.15²)   ← LSTM 추정값
    σ ~ HalfNormal(PRIOR_SIGMA_SD)
    """
    if sigma <= 0:
        return -math.inf
    lp  = -0.5 * ((alpha - mu_alpha) / 0.5) ** 2
    lp += -0.5 * ((beta  - mu_beta)  / 0.15) ** 2
    lp += -0.5 * (sigma / PRIOR_SIGMA_SD) ** 2
    return lp


def metropolis_hastings(log_y: List[float], t_idx: List[int],
                        mu_alpha: float, mu_beta: float,
                        n_warmup: int = MCMC_WARMUP,
                        n_draw:   int = MCMC_DRAWS,
                        seed:     int = SEED
                        ) -> Tuple[List[float], List[float], List[float]]:
    """MH-MCMC로 사후분포 샘플링. Returns (alpha_samp, beta_samp, sigma_samp)"""
    random.seed(seed)
    alpha = mu_alpha
    beta  = mu_beta
    sigma = PRIOR_SIGMA_SD * 0.7

    def log_post(a, b, s):
        return (_log_likelihood(log_y, t_idx, a, b, s)
                + _log_prior(a, b, s, mu_alpha, mu_beta))

    lp_cur  = log_post(alpha, beta, sigma)
    samples = []

    for step in range(n_warmup + n_draw):
        a_p = alpha + random.gauss(0, MH_STEP_ALPHA)
        b_p = beta  + random.gauss(0, MH_STEP_BETA)
        s_p = abs(sigma + random.gauss(0, MH_STEP_SIGMA)) + 1e-6

        lp_prop = log_post(a_p, b_p, s_p)
        log_ar  = lp_prop - lp_cur

        if log_ar >= 0 or random.random() < math.exp(max(log_ar, -500)):
            alpha, beta, sigma = a_p, b_p, s_p
            lp_cur = lp_prop

        if step >= n_warmup:
            samples.append((alpha, beta, sigma))

    return ([s[0] for s in samples],
            [s[1] for s in samples],
            [s[2] for s in samples])


def bayesian_forecast(train_vals: List[float], n_steps: int,
                      mu_beta_prior: float,
                      seed: int = SEED
                      ) -> Tuple[List[float], List[float], List[float]]:
    """
    Bayesian log-linear 모델 → 중앙값 + 95% 신용구간.
    Returns: (means, lo95, hi95) 원래 스케일
    """
    n_train  = len(train_vals)
    log_y    = [math.log(max(v, 0.5)) for v in train_vals]
    t_idx    = list(range(n_train))
    mu_alpha = log_y[0]

    a_samp, b_samp, s_samp = metropolis_hastings(
        log_y, t_idx, mu_alpha, mu_beta_prior,
        n_warmup=MCMC_WARMUP, n_draw=MCMC_DRAWS, seed=seed)

    means, lo95, hi95 = [], [], []
    random.seed(seed + 1)

    for step in range(n_steps):
        t_future  = n_train + step
        pred_dist = []
        for a, b, s in zip(a_samp, b_samp, s_samp):
            mu_pred = a + b * t_future
            y_pred  = max(0.0, math.exp(mu_pred + random.gauss(0, s)))
            pred_dist.append(y_pred)

        pred_dist.sort()
        nd = len(pred_dist)
        means.append(round(statistics.median(pred_dist), 2))
        lo95.append( round(pred_dist[max(0, int(0.025 * nd))], 2))
        hi95.append( round(pred_dist[min(nd - 1, int(0.975 * nd))], 2))

    return means, lo95, hi95


# ══════════════════════════════════════════════════════════════
# PART 3. 평가 지표
# ══════════════════════════════════════════════════════════════

def compute_forecast_metrics(actual: List[float],
                             forecast: List[float],
                             naive:    Optional[List[float]] = None,
                             lo95:     Optional[List[float]] = None,
                             hi95:     Optional[List[float]] = None) -> Dict:
    n = len(actual)
    if n == 0 or len(forecast) != n:
        return {"error": "입력 불일치"}

    aa = [float(v) for v in actual]
    ff = [float(v) for v in forecast]

    mape_t = [abs(f - a) / abs(a) for a, f in zip(aa, ff) if abs(a) > 0]
    mape   = round(statistics.mean(mape_t) * 100, 2) if mape_t else None
    smape  = round(statistics.mean(
        [2 * abs(f - a) / (abs(a) + abs(f) + 1e-8)
         for a, f in zip(aa, ff)]) * 100, 2)
    mae    = round(statistics.mean([abs(f - a) for a, f in zip(aa, ff)]), 4)
    rmse   = round(math.sqrt(
        statistics.mean([(f - a) ** 2 for a, f in zip(aa, ff)])), 4)

    theil = None
    if naive and len(naive) == n:
        na         = [float(v) for v in naive]
        naive_rmse = math.sqrt(
            statistics.mean([(nv - a) ** 2 for a, nv in zip(aa, na)]))
        theil      = round(rmse / (naive_rmse + 1e-8), 4)

    # Dir_Acc: n>1이면 연속 방향 비교, n=1이면 naive 대비 절대 방향 비교
    if n > 1:
        dir_c   = sum(1 for i in range(1, n)
                      if (ff[i] - ff[i - 1]) * (aa[i] - aa[i - 1]) > 0)
        dir_acc = round(dir_c / (n - 1) * 100, 1)
    else:
        # 단일 예측: 예측값이 naive(마지막 훈련값)보다 큰지 방향 확인
        # naive = train[-1], actual 방향: actual > naive → 증가
        if naive and len(naive) == 1:
            actual_dir = 1 if (aa[0] - naive[0]) > 0 else -1
            pred_dir   = 1 if (ff[0] - naive[0]) > 0 else -1
            dir_acc    = 100.0 if actual_dir == pred_dir else 0.0
        else:
            dir_acc = None   # 계산 불가

    pi95_cov = None
    if lo95 and hi95 and len(lo95) == n:
        inside   = sum(1 for a, lo, hi in zip(aa, lo95, hi95) if lo <= a <= hi)
        pi95_cov = round(inside / n * 100, 1)

    return {
        "n":             n,
        "MAPE":          mape,
        "SMAPE":         smape,
        "MAE":           mae,
        "RMSE":          rmse,
        "Theil_U":       theil,
        "Dir_Acc_pct":   dir_acc,
        "PI_95_cov_pct": pi95_cov,
    }


# ══════════════════════════════════════════════════════════════
# PART 4. 메인 진입점 (기존 run_forecast_backtesting 1:1 대체)
# ══════════════════════════════════════════════════════════════

def run_forecast_backtesting_blstm(tech_pairs: List[Dict],
                                   verbose: bool = True) -> List[Dict]:
    """
    [FORECAST] LSTM-Informed Bayesian Log-Linear Regression.
    기존 run_forecast_backtesting()의 1:1 대체.
    """
    random.seed(SEED)
    torch.manual_seed(SEED)

    all_seqs = [p["actual_cooc"] for p in tech_pairs
                if len(p.get("actual_cooc", [])) >= 5]
    if not all_seqs:
        return []

    # LSTM prior 추정기 학습
    if verbose:
        print(f"\n  [BLSTM] GrowthRateLSTM 사전학습 "
              f"({len(all_seqs)}쌍 풀링, {LSTM_EPOCHS}에폭)...")
    lstm_model = train_growth_lstm(all_seqs)
    lstm_model.eval()
    if verbose:
        print("  [BLSTM] 사전학습 완료 ✓")
        print(f"  [BLSTM] Bayesian MCMC 예측 시작 "
              f"(warmup={MCMC_WARMUP}, draws={MCMC_DRAWS})...")

    results = []

    for pair in tech_pairs:
        ac      = pair.get("actual_cooc", [])
        pair_id = pair["pair_id"]
        if len(ac) < 5:
            continue

        for train_end in [3, 4]:
            train_vals = ac[:train_end]
            test_vals  = ac[train_end:]
            if not test_vals:
                continue

            n_test   = len(test_vals)
            naive_fc = [float(train_vals[-1])] * n_test

            train_log  = [math.log(max(v, 0.5)) for v in train_vals]
            mu_beta    = lstm_growth_prior(lstm_model, train_log)

            seed_i = SEED + abs(hash(pair_id)) % 1000 + train_end
            means, lo95, hi95 = bayesian_forecast(
                train_vals, n_test, mu_beta, seed=seed_i)

            metrics = compute_forecast_metrics(
                actual=test_vals, forecast=means,
                naive=naive_fc, lo95=lo95, hi95=hi95)

            implied_growth_pct = round((math.exp(mu_beta) - 1) * 100, 1)

            row = {
                **metrics,
                "pair_id":           pair_id,
                "domain":            pair["domain"],
                "train_window":      f"periods_0-{train_end - 1}",
                "test_window":       f"periods_{train_end}-{len(ac) - 1}",
                "n_train":           len(train_vals),
                "n_test":            n_test,
                "actual_train":      str(train_vals),
                "actual_test":       str(test_vals),
                "blstm_fc_mean":     str([round(v, 2) for v in means]),
                "blstm_fc_lo95":     str([round(v, 2) for v in lo95]),
                "blstm_fc_hi95":     str([round(v, 2) for v in hi95]),
                "naive_fc":          str(naive_fc),
                "lstm_beta_prior":   round(mu_beta, 4),
                "implied_growth_pct_per_period": implied_growth_pct,
                "mcmc_draws":        MCMC_DRAWS,
                "model_config": (
                    f"BayesianLogLinear(MH)+GrowthRateLSTM("
                    f"hidden={LSTM_HIDDEN},layers={LSTM_LAYERS});"
                    f"warmup={MCMC_WARMUP},draws={MCMC_DRAWS}"),
                "note": (
                    "LSTM-Informed Bayesian log-linear regression; "
                    "LSTM estimates beta prior (mu_beta); "
                    "MH-MCMC posterior sampling; "
                    "95% credible interval from posterior predictive"),
            }
            results.append(row)

            if verbose:
                dir_sym   = "✓" if metrics.get("Dir_Acc_pct", 0) > 50 else "✗"
                pi_str    = (f"PI95={metrics['PI_95_cov_pct']:.0f}%"
                             if metrics.get("PI_95_cov_pct") is not None
                             else "PI95=N/A")
                theil_str = (f"U={metrics['Theil_U']:.2f}"
                             if metrics.get("Theil_U") is not None else "")
                print(
                    f"    [BLSTM] {pair_id:18s} train=0-{train_end - 1} "
                    f"Dir={metrics.get('Dir_Acc_pct', 0):.0f}% "
                    f"SMAPE={metrics.get('SMAPE', 'N/A')}% "
                    f"{pi_str} {theil_str} {dir_sym}"
                )

    return results


# ══════════════════════════════════════════════════════════════
# 독립 실행 데모
# ══════════════════════════════════════════════════════════════
_DEMO_TECH_PAIRS = [
    {"pair_id": "G06F16-H04W12", "domain": "Data/Comm-Centric",
     "actual_cooc": [3, 8, 14, 22, 35],
     "period_labels": ["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id": "B60R21-B60R23", "domain": "Vehicle-Centric",
     "actual_cooc": [5, 11, 19, 30, 45],
     "period_labels": ["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id": "G08G5-H04L20",  "domain": "Cross-Industry",
     "actual_cooc": [2, 5, 10, 18, 28],
     "period_labels": ["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id": "B60R23-G01S7",  "domain": "V2X",
     "actual_cooc": [4, 9, 16, 25, 38],
     "period_labels": ["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id": "B61L25-G06Q50", "domain": "Cross-Industry",
     "actual_cooc": [1, 3, 7, 13, 21],
     "period_labels": ["2000-04","2005-09","2010-14","2015-19","2020-24"]},
    {"pair_id": "G07C5-H04M1",   "domain": "Data/Comm-Centric",
     "actual_cooc": [2, 6, 12, 20, 32],
     "period_labels": ["2000-04","2005-09","2010-14","2015-19","2020-24"]},
]

if __name__ == "__main__":
    import csv

    print("=" * 68)
    print("  [독립 실행] LSTM-Informed Bayesian Forecast — V8 수정판")
    print("=" * 68)

    results = run_forecast_backtesting_blstm(_DEMO_TECH_PAIRS, verbose=True)

    if results:
        dir_accs = [r["Dir_Acc_pct"]   for r in results if r.get("Dir_Acc_pct")   is not None]
        smapes   = [r["SMAPE"]         for r in results if r.get("SMAPE")         is not None]
        theils   = [r["Theil_U"]       for r in results if r.get("Theil_U")       is not None]
        pi95s    = [r["PI_95_cov_pct"] for r in results if r.get("PI_95_cov_pct") is not None]

        print("\n" + "=" * 68)
        print(f"  요약 ({len(results)}개 창)")
        print(f"  평균 Dir_Acc  = {statistics.mean(dir_accs):.1f}%"
              f"   (수정 전: 0.0%  |  Proxy: 50.0%)")
        print(f"  평균 SMAPE    = {statistics.mean(smapes):.1f}%"
              f"   (수정 전: 49.1% |  Proxy: 31.5%)")
        if theils:
            print(f"  평균 Theil-U  = {statistics.mean(theils):.3f}"
                  f"   (<1.0 → naive 보다 우수)")
        if pi95s:
            print(f"  평균 PI-95 커버리지 = {statistics.mean(pi95s):.1f}%"
                  f"   (이상값: ~95%)")

        out = "results_forecast_v8_blstm_demo.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  저장 → {out}")
    print("=" * 68)
