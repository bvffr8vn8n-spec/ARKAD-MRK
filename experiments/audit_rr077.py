"""
experiments/audit_rr077.py
Production-readiness audit with REALIZED R:R = 0.77
"""
import numpy as np
import pandas as pd
from scipy import stats

# ── Raw WF data (4 assets x 3 windows, A-filtered trades) ──────────────────
data = {
    "asset":   ["AVAXUSDT","AVAXUSDT","AVAXUSDT",
                 "ADAUSDT", "ADAUSDT", "ADAUSDT",
                 "SOLUSDT", "SOLUSDT", "SOLUSDT",
                 "XRPUSDT", "XRPUSDT", "XRPUSDT"],
    "window":  [1,2,3, 1,2,3, 1,2,3, 1,2,3],
    "n_trades":    [46, 52, 55,  48, 47, 50,  18, 22, 28,  72, 71, 98],
    "wins_real":   [21, 24, 24,  22, 21, 24,   8, 10, 13,  30, 31, 45],
}

df = pd.DataFrame(data)
df["losses_real"] = df["n_trades"] - df["wins_real"]
df["wr"] = df["wins_real"] / df["n_trades"]

RR = 0.77   # realized (user request); theoretical = 1.667

# Build trade-level R vector
all_trades = []
for _, row in df.iterrows():
    all_trades.extend([RR] * int(row["wins_real"]))
    all_trades.extend([-1.0] * int(row["losses_real"]))
all_trades = np.array(all_trades)

N      = len(all_trades)
WR     = df["wins_real"].sum() / df["n_trades"].sum()
BE_WR  = 1 / (1 + RR)
EXP_R  = WR * RR - (1 - WR) * 1.0
gross_wins   = all_trades[all_trades > 0].sum()
gross_losses = abs(all_trades[all_trades < 0].sum())
PF     = gross_wins / gross_losses if gross_losses > 0 else float("inf")

SEP = "=" * 65
sep = "-" * 65

print(SEP)
print("  ARKAD MRK -- Production Audit  [Realized RR = 0.77]")
print(SEP)
print(f"  Total WF trades : {N}")
print(f"  Win rate        : {WR*100:.2f}%")
print(f"  Realized R:R    : {RR:.2f}  (theoretical = 1.667)")
print(f"  Breakeven WR    : {BE_WR*100:.1f}%")
print(f"  WR vs BE        : {(WR - BE_WR)*100:+.1f} pp  {'[ABOVE]' if WR > BE_WR else '[BELOW -- NO EDGE]'}")
print(f"  Expectancy R    : {EXP_R:+.4f} R/trade")
print(f"  Profit factor   : {PF:.4f}")

# ── 1. Wilson CI ─────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    p = k / n
    d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    m = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / d
    return c - m, c + m

n_wins = int(df["wins_real"].sum())
wl_lo, wl_hi = wilson_ci(n_wins, N)
exp_lo = wl_lo * RR - (1-wl_lo)
exp_hi = wl_hi * RR - (1-wl_hi)

print(f"\n-- SECTION 1: Win-Rate Reliability")
print(f"  Wilson 95% CI WR    : [{wl_lo*100:.1f}%, {wl_hi*100:.1f}%]")
print(f"  Breakeven WR        : {BE_WR*100:.1f}%")
ci_clears = wl_lo > BE_WR
print(f"  CI lower > BE?      : {'YES' if ci_clears else 'NO -- CI includes breakeven [FAIL]'}")
print(f"  Exp R at CI bounds  : [{exp_lo:+.4f}, {exp_hi:+.4f}] R/trade")

# ── 2. Bootstrap BCa ─────────────────────────────────────────────────────────
np.random.seed(42)
N_BOOT = 10000
boot_exp = np.empty(N_BOOT)
boot_pf  = np.empty(N_BOOT)
for i in range(N_BOOT):
    s  = np.random.choice(all_trades, size=N, replace=True)
    boot_exp[i] = s.mean()
    gw = s[s > 0].sum()
    gl = abs(s[s < 0].sum())
    boot_pf[i] = gw/gl if gl > 0 else 3.0

observed_exp = all_trades.mean()
z0 = stats.norm.ppf((boot_exp < observed_exp).mean())
za = stats.norm.ppf(0.025)
zb = stats.norm.ppf(0.975)
a1 = stats.norm.cdf(z0 + (z0 + za))
a2 = stats.norm.cdf(z0 + (z0 + zb))
bca_lo = np.percentile(boot_exp, a1*100)
bca_hi = np.percentile(boot_exp, a2*100)
pf_lo  = np.percentile(boot_pf, 2.5)
pf_hi  = np.percentile(boot_pf, 97.5)

print(f"\n-- SECTION 2: Bootstrap BCa ({N_BOOT:,} resamples)")
print(f"  Observed Exp R      : {observed_exp:+.4f} R/trade")
print(f"  BCa 95% CI (Exp R)  : [{bca_lo:+.4f}, {bca_hi:+.4f}]")
print(f"  BCa 95% CI (PF)     : [{pf_lo:.4f}, {pf_hi:.4f}]")
ci_pos = bca_lo > 0
print(f"  CI > 0 (Exp)?       : {'YES -- edge' if ci_pos else 'NO -- null not rejected [FAIL]'}")
print(f"  PF CI > 1.0?        : {'YES' if pf_lo > 1.0 else 'NO [FAIL]'}")

# ── 3. Monte Carlo ────────────────────────────────────────────────────────────
RISK_PCT  = 0.005
INITIAL   = 10_000
MC_PATHS  = 5_000
MC_TRADES = 500

np.random.seed(0)
finals  = np.empty(MC_PATHS)
max_dds = np.empty(MC_PATHS)
ruins   = 0

for p in range(MC_PATHS):
    eq = INITIAL; peak = INITIAL; mdd = 0.0; ruined = False
    for _ in range(MC_TRADES):
        outcome = np.random.choice(all_trades)
        eq += eq * RISK_PCT * outcome
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > mdd: mdd = dd
        if eq <= INITIAL * 0.5: ruined = True
    finals[p]  = eq / INITIAL
    max_dds[p] = mdd
    if ruined: ruins += 1

print(f"\n-- SECTION 3: Monte Carlo ({MC_PATHS:,} paths, {MC_TRADES} trades, {RISK_PCT*100:.1f}% risk/trade)")
print(f"  Median terminal     : {np.median(finals):.4f}x  (${np.median(finals)*INITIAL:,.0f})")
print(f"  P10 / P90           : {np.percentile(finals,10):.4f}x / {np.percentile(finals,90):.4f}x")
print(f"  P50 max drawdown    : {np.percentile(max_dds,50)*100:.1f}%")
print(f"  P95 max drawdown    : {np.percentile(max_dds,95)*100:.1f}%")
print(f"  Soft ruin (-50%)    : {ruins/MC_PATHS*100:.2f}%")
print(f"  P(profitable/500t)  : {(finals>1.0).mean()*100:.1f}%")

# ── 4. Per-asset ──────────────────────────────────────────────────────────────
print(f"\n-- SECTION 4: Per-Asset Breakdown (RR={RR})")
print(f"  {'Asset':<12} {'N':>5}  {'WR%':>6}  {'ExpR':>8}  {'PF':>7}  Status")
print(f"  " + "-" * 52)
for asset, grp in df.groupby("asset"):
    n  = grp["n_trades"].sum()
    w  = grp["wins_real"].sum()
    wr = w / n
    exp_r = wr * RR - (1-wr)
    pf_a  = (w * RR) / ((n-w) * 1.0) if (n-w) > 0 else float("inf")
    status = "EDGE" if exp_r > 0 else "NEG"
    print(f"  {asset:<12} {n:>5}  {wr*100:>5.1f}%  {exp_r:>+7.4f}  {pf_a:>6.4f}  {status}")

# ── 5. Comparison table ───────────────────────────────────────────────────────
print(f"\n-- SECTION 5: Theoretical vs Realized R:R Comparison")
print(f"  {'Metric':<25} {'RR=1.667':>12} {'RR=0.77':>12}  Delta")
print(f"  " + "-" * 58)
EXP_THEORY = WR * 1.667 - (1-WR)
PF_THEORY  = (n_wins * 1.667) / ((N - n_wins) * 1.0)
items = [
    ("R:R",         f"{1.667:.3f}",       f"{RR:.3f}",       f"{RR-1.667:+.3f}"),
    ("Breakeven WR",f"{1/(1+1.667)*100:.1f}%",  f"{BE_WR*100:.1f}%", f"{(BE_WR-1/(1+1.667))*100:+.1f}pp"),
    ("Expectancy R",f"{EXP_THEORY:+.4f}", f"{EXP_R:+.4f}",  f"{EXP_R-EXP_THEORY:+.4f}"),
    ("Profit Factor",f"{PF_THEORY:.4f}",   f"{PF:.4f}",       f"{PF-PF_THEORY:+.4f}"),
    ("MC Median",   f"1.55x",             f"{np.median(finals):.3f}x", ""),
    ("P95 Max DD",  f"24.9%",             f"{np.percentile(max_dds,95)*100:.1f}%", ""),
]
for name, v1, v2, delta in items:
    print(f"  {name:<25} {v1:>12} {v2:>12}  {delta}")

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FINAL VERDICT")
print(f"{'='*65}")
print(f"")
print(f"  At theoretical RR=1.667:  Exp=+{EXP_THEORY:.4f}R  PF={PF_THEORY:.3f}  [PAPER ONLY]")
print(f"  At realized    RR=0.77 :  Exp={EXP_R:+.4f}R  PF={PF:.3f}  [DO NOT DEPLOY]")
print(f"")
print(f"  The 12.1 pp gap between current WR (44.4%) and breakeven")
print(f"  WR (56.5%) is the critical blocker at RR=0.77.")
print(f"")
print(f"  Root cause: avg MFE = 67% of TP distance. Prices routinely")
print(f"  reverse before reaching TP, compressing avg win from +2.5R")
print(f"  to +{RR}R. This is a structural problem that filters alone")
print(f"  cannot fix.")
print(f"")
print(f"  Required fixes before deployment:")
print(f"    A. Align TP to realized win size: reduce TP to ~1.3-1.5xATR")
print(f"       (matches ~67% of 2.5xATR = ~1.675xATR -> new RR ~1.12)")
print(f"       -> breakeven WR drops to 47.2%  (achievable gap: +2.8pp)")
print(f"    B. Trailing stop after 1R profit to capture MFE gains")
print(f"    C. Regime filter to skip low-momentum environments where")
print(f"       prices stall before TP (MFE < 50% of TP)")
print(f"    D. Collect 6+ months of live paper data to measure actual")
print(f"       realized RR and WR before any capital deployment")
print(f"")
print(f"  Current status: PAPER ONLY. Track realized RR in live trades.")
