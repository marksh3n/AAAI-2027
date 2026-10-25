# policies and their cost, per attempt. pure functions on tables, no I/O; analyze.py does the rest.
# an attempt is one source trace: one run of a policy on its problem. its outcome is 1 = correct.
# cost is total generated tokens of the attempt, everything counted: tokens later discarded, every
# replacement, every extra vote, and any rollouts used to decide (selection cost).
#   none        keeps the trace: correct_i, tokens n_i
#   restart     at checkpoint c, for a flagged LIVE trace: discard it, sample afresh from the prompt.
#               expected outcome r0 (the problem's t0 bank), tokens c + L0 (generated, then discarded,
#               then the new sample)
#   backtrack   at c, back to c' = BACKTRACK_TO[c]: continue the trace's own prefix at c' afresh.
#               outcome r(c'), tokens c + L(c') (c' tokens kept, c - c' discarded, then the new tail)
#   resample    (validity check, not an arm) continue the trace's own prefix at c afresh: outcome r(c),
#               tokens c + L(c). sources and bank come from one engine, so its effect must be ~0
#   aggregate   m extra fresh samples and a plurality vote over the trace's answer and theirs.
#               outcome A_m (exact over every m-subset of the problem's t0 bank), tokens n_i + m * L0
# a replacement's outcome and length are the mean over its bank: the expected value of drawing one
# replacement, unbiased because the policy decides from the source prefix alone and the bank is
# sampled independently by the same engine. flag weights w in [0, 1]; a random policy is w = pi.
from itertools import combinations
import numpy as np


def plurality_correct(keys, correct_key="C"):
    # P(the vote picks the correct answer): plurality over non-None keys, ties split uniformly
    votes = {}
    for k in keys:
        if k is not None:
            votes[k] = votes.get(k, 0) + 1
    if not votes:
        return 0.0
    top = max(votes.values())
    tied = [k for k, v in votes.items() if v == top]
    return (correct_key in tied) / len(tied)


def aggregate_acc(source_key, bank_keys, m):
    # exact expectation over every m-subset of the bank, drawn without replacement
    subsets = list(combinations(range(len(bank_keys)), m))
    if not subsets:
        raise ValueError(f"bank of {len(bank_keys)} cannot supply {m} extra votes")
    return float(np.mean([plurality_correct([source_key] + [bank_keys[j] for j in s]) for s in subsets]))


def arm_effect(t, action, c):
    """per attempt, if flagged: gain in outcome and extra tokens over keeping it. t holds correct, n,
    live, r0, L0 and, for backtrack, r_prev, L_prev. not-live attempts get 0 / 0: never flaggable."""
    if action == "restart":
        r, L = t.r0.values, t.L0.values
    elif action == "resample":  # validity check only: a fresh continuation of the same prefix
        r, L = t.r.values, t.L.values
    elif action == "backtrack":
        r, L = t.r_prev.values, t.L_prev.values
    else:
        raise ValueError(action)
    live = t.live.values.astype(bool)
    gain = np.where(live, r - t.correct.values, 0.0)
    extra = np.where(live, c + L - t.n.values, 0.0)
    if np.isnan(gain).any() or np.isnan(extra).any():
        raise ValueError(f"{action} at {c}: a live attempt has no bank to draw its replacement from")
    return gain, extra


def flags(score, tau, live):
    # strictly above the frozen threshold, live attempts only
    return (np.asarray(score, float) > tau) & np.asarray(live, bool)


class Sums:
    """per-problem sums under bootstrap draws: draws (B, P) multiplicities of the problems in
    `problems` order, attempts mapped to them by problem_id. every arm shares the draws."""

    def __init__(self, problem_ids, draws):
        self.pids, self.col = np.unique(np.asarray(problem_ids), return_inverse=True)
        self.draws = np.vstack([np.ones(len(self.pids)), draws])  # row 0 = the point estimate
        self.N = self.draws @ np.bincount(self.col, minlength=len(self.pids)).astype(float)

    def mean(self, v):
        # (B + 1,) attempt-weighted means, row 0 the estimate
        return self.draws @ np.bincount(self.col, np.asarray(v, float), minlength=len(self.pids)) / self.N

    def total(self, v):
        return self.draws @ np.bincount(self.col, np.asarray(v, float), minlength=len(self.pids))


def matched(sums, base_acc, base_tok, pol_acc, pol_tok, comp_gain, comp_extra):
    """a comparator whose spend is linear in its rate pi (random flagging, aggregation), funded to the
    policy's total tokens in every draw: pi = (policy extra tokens) / (comparator extra tokens at pi=1).
    returns per draw (comparator accuracy, pi, status): status 0 matched, 1 pi > 1 (comparator capped
    at pi = 1, cannot spend that much), -1 pi < 0 (no non-negative rate spends what the policy does;
    the comparator falls back to no intervention, i.e. pi = 0)."""
    spend = sums.total(np.asarray(pol_tok) - base_tok)
    full = sums.total(comp_extra)
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = np.where(full != 0, spend / full, np.where(spend == 0, 0.0, np.nan))
    status = np.where(np.isnan(pi) | (pi < 0), -1, np.where(pi > 1, 1, 0))
    pi = np.clip(np.nan_to_num(pi, nan=0.0), 0.0, 1.0)
    acc = sums.mean(base_acc) + pi * sums.total(comp_gain) / sums.N
    return acc, pi, status


def oracle_restart(t, q, k_sel, c):
    """rollout-ranked restart, cross-fitted on the prefix bank's two halves: half h ranks (low r first),
    the decision costs k_sel continuations of every live attempt (selection cost), the replacement is
    the t0 bank. a diagnostic for the diagnostic gap, never a policy under test. returns acc, tok."""
    live = t.live.values.astype(bool)
    gain, extra = arm_effect(t, "restart", c)
    acc, tok = np.zeros(len(t)), np.zeros(len(t))
    for h in (0, 1):
        s = np.where(live, 1 - t[f"r_h{h}"].values, -np.inf)
        w = top_share(s, q * live.sum())
        sel = np.where(live, k_sel * t[f"L_h{h}"].values, 0.0)
        acc += (t.correct.values + w * gain) / 2
        tok += (t.n.values + sel + w * extra) / 2
    return acc, tok


def top_share(score, budget):
    # flag `budget` rows, highest score first; a tie group straddling it is flagged fractionally
    score = np.asarray(score, float)
    w, done = np.zeros(len(score)), 0.0
    for v in np.unique(score[np.isfinite(score)])[::-1]:
        idx = np.flatnonzero(score == v)
        take = min(len(idx), budget - done)
        if take <= 0:
            break
        w[idx] = take / len(idx)
        done += take
    return w
