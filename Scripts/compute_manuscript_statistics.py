"""
Compute all manuscript statistics after excluding experiment 1 participants
with more than 18 trials. Writes results into the "New" columns of
Manuscript_statistics_extraction.xlsx without touching existing content.
"""

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import os, sys, warnings
from itertools import takewhile

warnings.filterwarnings('ignore')

ROOT = '/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush'
sys.path.insert(0, os.path.join(ROOT, 'Scripts'))
from modules.params import *
from modules.functions import str2int_dataset, dl_distance, compare_tokens, array_structure

# ──────────────────────────────────────────────
# 1.  DATA LOADING
# ──────────────────────────────────────────────

df1_raw = pd.read_csv(os.path.join(ROOT, 'Data/processed/experiment-1/experiment1_processed_10h05_31032023_data.csv'))

# Exclude participants with >18 trials
trial_counts = df1_raw.groupby('participant_ID').size()
excluded_ids = set(trial_counts[trial_counts > 18].index)
df1 = df1_raw[~df1_raw['participant_ID'].isin(excluded_ids)].copy().reset_index(drop=True)
df1 = df1[df1['seq_name'] != 'Training'].reset_index(drop=True)
df1 = str2int_dataset(df1, processed=True, label_int_col=label_int_col)

df2 = pd.read_csv(os.path.join(ROOT, 'Data/processed/experiment-2/processed_20240416_15h50_memocrush_extension_pilote2_data.csv'))
df2 = df2[df2['state'] == 'main_experiment'].reset_index(drop=True)
df2 = str2int_dataset(df2, processed=True, label_int_col=label_int_col)

N1 = df1['participant_ID'].nunique()
N2 = df2['participant_ID'].nunique()
print(f"Exp1 N = {N1}  (excluded {len(excluded_ids)} participants)")
print(f"Exp2 N = {N2}")

# ──────────────────────────────────────────────
# 2.  HELPER FUNCTIONS
# ──────────────────────────────────────────────

def _r(x, y, p):
    """Effect size r = Z/sqrt(N_nonzero) for Wilcoxon signed-rank."""
    nz = int(np.sum(np.array(x) - np.array(y) != 0))
    if nz == 0:
        return np.nan
    z = scipy_stats.norm.isf(p / 2)
    return round(z / np.sqrt(nz), 4)

def wilcoxon_pair(a, b):
    """Return (W, p, r) for matched arrays a, b."""
    stat, p = scipy_stats.wilcoxon(a, b)
    r = _r(a, b, p)
    return round(stat, 4), round(p, 6), r

def per_participant_error_rate(data, seq_name):
    """Per-participant mean error rate (%) for a given sequence name."""
    rows = []
    for pid in data['participant_ID'].unique():
        sub = data[(data['participant_ID'] == pid) & (data['seq_name'] == seq_name)]
        if len(sub) > 0:
            rows.append((pid, 100 * (sub['performance'] != 'success').mean()))
    return pd.DataFrame(rows, columns=['pid', 'er']).set_index('pid')['er']

def per_participant_dl(data, seq_name):
    """Per-participant mean DL distance for a given sequence name."""
    rows = []
    for pid in data['participant_ID'].unique():
        sub = data[(data['participant_ID'] == pid) & (data['seq_name'] == seq_name)]
        if len(sub) > 0:
            rows.append((pid, sub['distance_dl'].mean()))
    return pd.DataFrame(rows, columns=['pid', 'dl']).set_index('pid')['dl']

def per_participant_rt(data, seq_name):
    """Per-participant mean total RT (ms) = sum of interclick_time per trial."""
    rows = []
    for pid in data['participant_ID'].unique():
        sub = data[(data['participant_ID'] == pid) & (data['seq_name'] == seq_name)]
        if len(sub) > 0:
            rts = [sum(t) for t in sub['interclick_time']]
            rows.append((pid, np.mean(rts)))
    return pd.DataFrame(rows, columns=['pid', 'rt']).set_index('pid')['rt']

def group_mean_sem(series_dict, seq_list):
    """Mean and SEM of per-participant means pooled across multiple sequences."""
    all_pids = set.intersection(*[set(series_dict[s].index) for s in seq_list])
    vals = [np.mean([series_dict[s].loc[pid] for s in seq_list]) for pid in all_pids]
    return np.mean(vals), scipy_stats.sem(vals), vals

# ──────────────────────────────────────────────
# 3.  EXP 1 — STATISTICS
# ──────────────────────────────────────────────

stats = {}   # key: row_id, value: dict of new values

# ── Row 1: Spatial position accuracy = 1 - TokenErr rate ──────
# Sequences are all length 12; TokenErr flags trials where the response
# uses a different set of spatial positions than the target sequence.
pp_pos_acc = []
for pid in df1['participant_ID'].unique():
    sub = df1[df1['participant_ID'] == pid]
    pp_pos_acc.append(100 * (1 - sub['TokenErr'].mean()))
stats[1] = {'New N': N1, 'New value': round(np.mean(pp_pos_acc), 2)}
print(f"Row 1  spatial_position_acc = {stats[1]['New value']}%")

# ── Row 2: Correct temporal order (performance==success across all trials) ──
pp_success = []
for pid in df1['participant_ID'].unique():
    sub = df1[df1['participant_ID'] == pid]
    pp_success.append(100 * (sub['performance'] == 'success').mean())
stats[2] = {'New N': N1,
            'New value': round(np.mean(pp_success), 2),
            'New SEM':   round(scipy_stats.sem(pp_success), 2)}
print(f"Row 2  correct_temporal_order = {stats[2]['New value']} ± {stats[2]['New SEM']}%")

# ── Rows 3-5: Primacy effects ──────────────────────────────────
pp_primacy = []   # shape (N, 12) — per-position accuracy per participant
for pid in df1['participant_ID'].unique():
    sub = df1[df1['participant_ID'] == pid]
    pos_counts = np.zeros(12)
    for _, row in sub.iterrows():
        s, r = row['seq'], row['sequences_response']
        n = min(len(s), len(r), 12)
        for i in range(n):
            pos_counts[i] += (s[i] == r[i])
    pp_primacy.append(100 * pos_counts / len(sub))

pp_primacy = np.array(pp_primacy)   # (N, 12)
mean_pos_acc = pp_primacy.mean(axis=0)  # per-position
overall_mean = pp_primacy.mean()        # grand mean across all positions
above_mean = mean_pos_acc - overall_mean

for k, row_id in enumerate([3, 4, 5]):
    stats[row_id] = {'New N': N1, 'New value': round(above_mean[k], 2)}
    print(f"Row {row_id}  primacy_item{k+1} = {stats[row_id]['New value']}% pts above mean")

# ── Rows 6-11: Forward span ──────────────────────────────────
def _accurate_length(seq, response):
    return sum(1 for _ in takewhile(lambda p: p[0] == p[1], zip(seq, response)))

df1['accurate_length'] = [_accurate_length(r['seq'], r['sequences_response'])
                          for _, r in df1.iterrows()]

span_seqs = {
    6:  'control Repetition-3',
    7:  'control Repetition-4',
    9:  'Repetition-3',
    10: 'Repetition-4',
}

pp_span = {}
for row_id, sname in span_seqs.items():
    s = df1[df1['seq_name'] == sname].groupby('participant_ID')['accurate_length'].mean()
    mean_s = round(s.mean(), 2)
    sem_s  = round(scipy_stats.sem(s), 2)
    stats[row_id] = {'New N': N1, 'New value': mean_s, 'New SEM': sem_s}
    pp_span[sname] = s
    print(f"Row {row_id}  forward_span [{sname}] = {mean_s} ± {sem_s}")

# Row 8: CRep-3 vs CRep-4 Wilcoxon
c3 = pp_span['control Repetition-3']
c4 = pp_span['control Repetition-4']
common = c3.index.intersection(c4.index)
W8, p8, r8 = wilcoxon_pair(c3.loc[common].values, c4.loc[common].values)
stats[8] = {'New N': N1, 'New statistic': W8, 'New p': round(p8, 4)}
print(f"Row 8  CRep-3 vs CRep-4: W={W8}, p={p8:.4f}")

# Row 11: Rep-3 vs Rep-4 Wilcoxon
r3 = pp_span['Repetition-3']
r4 = pp_span['Repetition-4']
common = r3.index.intersection(r4.index)
W11, p11, r11 = wilcoxon_pair(r3.loc[common].values, r4.loc[common].values)
stats[11] = {'New N': N1, 'New statistic': W11, 'New p': round(p11, 4)}
print(f"Row 11 Rep-3 vs Rep-4: W={W11}, p={p11:.4f}")

# ── Rows 12-14: Error rate (structured Rep-2/3/4 vs control) ──
struc_seqs = ['Repetition-2', 'Repetition-3', 'Repetition-4']
ctrl_seqs  = ['control Repetition-2', 'control Repetition-3', 'control Repetition-4']

er_s = {s: per_participant_error_rate(df1, s) for s in struc_seqs}
er_c = {s: per_participant_error_rate(df1, s) for s in ctrl_seqs}

mean_er_s, sem_er_s, pp_er_s_vals = group_mean_sem(er_s, struc_seqs)
mean_er_c, sem_er_c, pp_er_c_vals = group_mean_sem(er_c, ctrl_seqs)

stats[12] = {'New N': N1, 'New value': round(mean_er_s, 2), 'New SEM': round(sem_er_s, 2)}
stats[13] = {'New N': N1, 'New value': round(mean_er_c, 2), 'New SEM': round(sem_er_c, 2)}
print(f"Row 12 error_rate structured = {stats[12]['New value']} ± {stats[12]['New SEM']}%")
print(f"Row 13 error_rate control    = {stats[13]['New value']} ± {stats[13]['New SEM']}%")

W14, p14, r14 = wilcoxon_pair(pp_er_s_vals, pp_er_c_vals)
stats[14] = {'New N': N1, 'New statistic': W14, 'New p': round(p14, 6),
             'New effect size': r14}
print(f"Row 14 structured vs control ER: W={W14}, p={p14:.6f}, r={r14}")

# ── Rows 15-17: Edit distance ──────────────────────────────────
dl_s = {s: per_participant_dl(df1, s) for s in struc_seqs}
dl_c = {s: per_participant_dl(df1, s) for s in ctrl_seqs}

mean_dl_s, sem_dl_s, pp_dl_s_vals = group_mean_sem(dl_s, struc_seqs)
mean_dl_c, sem_dl_c, pp_dl_c_vals = group_mean_sem(dl_c, ctrl_seqs)

stats[15] = {'New N': N1, 'New value': round(mean_dl_s, 2), 'New SEM': round(sem_dl_s, 2)}
stats[16] = {'New N': N1, 'New value': round(mean_dl_c, 2), 'New SEM': round(sem_dl_c, 2)}
print(f"Row 15 edit_dist structured = {stats[15]['New value']} ± {stats[15]['New SEM']}")
print(f"Row 16 edit_dist control    = {stats[16]['New value']} ± {stats[16]['New SEM']}")

W17, p17, r17 = wilcoxon_pair(pp_dl_s_vals, pp_dl_c_vals)
stats[17] = {'New N': N1, 'New statistic': W17, 'New p': round(p17, 6), 'New effect size': r17}
print(f"Row 17 structured vs control DL: W={W17}, p={p17:.6f}, r={r17}")

# ── Rows 18-23: Paired contrasts ──────────────────────────────
all_paired_seqs = struc_seqs + ctrl_seqs + ['Repetition-Nested',
                  'control NoLocal nested', 'control NoGlobal nested']
er_all = {s: per_participant_error_rate(df1, s) for s in all_paired_seqs}
dl_all = {s: per_participant_dl(df1, s) for s in all_paired_seqs}

def paired_wilcoxon(measure_dict, seq_a, seq_b):
    common = measure_dict[seq_a].index.intersection(measure_dict[seq_b].index)
    a, b = measure_dict[seq_a].loc[common].values, measure_dict[seq_b].loc[common].values
    W, p, r = wilcoxon_pair(a, b)
    return W, p, r

# Row 18: each structured vs paired control (ER + DL), report minimum r across 6 pairs
row18_r = []
for sa, ca in zip(struc_seqs, ctrl_seqs):
    for measure in [er_all, dl_all]:
        _, p, r = paired_wilcoxon(measure, sa, ca)
        row18_r.append(r)
min_r18 = round(min(row18_r), 4)
stats[18] = {'New N': N1, 'New effect size': f'>= {min_r18}',
             'New p': '< .001'}
print(f"Row 18 all paired contrasts ER+DL: min r={min_r18}")

# Row 19: ER Rep-Nested vs Rep-Local (=control NoGlobal nested)
W19, p19, r19 = paired_wilcoxon(er_all, 'Repetition-Nested', 'control NoGlobal nested')
stats[19] = {'New N': N1, 'New statistic': W19, 'New p': round(p19, 6), 'New effect size': r19}
print(f"Row 19 ER Rep-Nested vs Rep-Local: W={W19}, p={p19:.6f}, r={r19}")

# Row 20: DL Rep-Nested vs Rep-Local
W20, p20, r20 = paired_wilcoxon(dl_all, 'Repetition-Nested', 'control NoGlobal nested')
stats[20] = {'New N': N1, 'New statistic': W20, 'New p': round(p20, 6), 'New effect size': r20}
print(f"Row 20 DL Rep-Nested vs Rep-Local: W={W20}, p={p20:.6f}, r={r20}")

# Row 21: ER+DL Rep-Nested vs Rep-Global (=control NoLocal nested) — report min r
W21e, p21e, r21e = paired_wilcoxon(er_all, 'Repetition-Nested', 'control NoLocal nested')
W21d, p21d, r21d = paired_wilcoxon(dl_all, 'Repetition-Nested', 'control NoLocal nested')
min_r21 = round(min(r21e, r21d), 4)
stats[21] = {'New N': N1, 'New effect size': f'> {min_r21}', 'New p': '< .001'}
print(f"Row 21 ER+DL Rep-Nested vs Rep-Global: r_ER={r21e}, r_DL={r21d}")

# Row 22: ER+DL Rep-Nested vs CRep-3
W22e, p22e, r22e = paired_wilcoxon(er_all, 'Repetition-Nested', 'control Repetition-3')
W22d, p22d, r22d = paired_wilcoxon(dl_all, 'Repetition-Nested', 'control Repetition-3')
min_r22 = round(min(r22e, r22d), 4)
stats[22] = {'New N': N1, 'New effect size': f'> {min_r22}', 'New p': '< .001'}
print(f"Row 22 ER+DL Rep-Nested vs CRep-3: r_ER={r22e}, r_DL={r22d}")

# Row 23: ER+DL Rep-Nested vs Rep-3 (null expected)
W23e, p23e, r23e = paired_wilcoxon(er_all, 'Repetition-Nested', 'Repetition-3')
W23d, p23d, r23d = paired_wilcoxon(dl_all, 'Repetition-Nested', 'Repetition-3')
stats[23] = {'New N': N1,
             'New p': f'{max(p23e, p23d):.4f}',
             'New statistic': W23e}
print(f"Row 23 ER+DL Rep-Nested vs Rep-3: p_ER={p23e:.4f}, p_DL={p23d:.4f}")

# ── Rows 24-33: Response time ──────────────────────────────────
rt_seqs_all = (struc_seqs + ctrl_seqs +
               ['Repetition-Nested', 'control NoLocal nested', 'control NoGlobal nested',
                'control Repetition-3'])
rt_all = {s: per_participant_rt(df1, s) for s in rt_seqs_all}

mean_rt_s, sem_rt_s, pp_rt_s = group_mean_sem(rt_all, struc_seqs)
mean_rt_c, sem_rt_c, pp_rt_c = group_mean_sem(rt_all, ctrl_seqs)

stats[24] = {'New N': N1, 'New value': round(mean_rt_s), 'New SEM': round(sem_rt_s)}
stats[25] = {'New N': N1, 'New value': round(mean_rt_c), 'New SEM': round(sem_rt_c)}
print(f"Row 24 RT structured = {stats[24]['New value']} ± {stats[24]['New SEM']} ms")
print(f"Row 25 RT control    = {stats[25]['New value']} ± {stats[25]['New SEM']} ms")

W26, p26, r26 = wilcoxon_pair(pp_rt_s, pp_rt_c)
stats[26] = {'New N': N1, 'New statistic': W26, 'New p': round(p26, 4), 'New effect size': r26}
print(f"Row 26 RT structured vs control: W={W26}, p={p26:.4f}, r={r26}")

for row_id, (sa, ca) in zip([27, 28, 29], zip(struc_seqs, ctrl_seqs)):
    common = rt_all[sa].index.intersection(rt_all[ca].index)
    W, p, r = wilcoxon_pair(rt_all[sa].loc[common].values, rt_all[ca].loc[common].values)
    stats[row_id] = {'New N': N1, 'New statistic': W, 'New p': round(p, 4), 'New effect size': r}
    print(f"Row {row_id} RT {sa} vs {ca}: W={W}, p={p:.4f}, r={r}")

nested_rt = rt_all['Repetition-Nested']
local_rt  = rt_all['control NoGlobal nested']   # = Rep-Local
global_rt = rt_all['control NoLocal nested']    # = Rep-Global
crep3_rt  = rt_all['control Repetition-3']
rep3_rt   = rt_all['Repetition-3']

stats[30] = {'New N': N1, 'New value': round(nested_rt.mean()), 'New SEM': round(scipy_stats.sem(nested_rt))}
stats[31] = {'New N': N1, 'New value': round(local_rt.mean()),  'New SEM': round(scipy_stats.sem(local_rt))}
print(f"Row 30 RT Rep-Nested = {stats[30]['New value']} ± {stats[30]['New SEM']} ms")
print(f"Row 31 RT Rep-Local  = {stats[31]['New value']} ± {stats[31]['New SEM']} ms")

# Row 32: Rep-Nested vs Rep-3 / CRep-3 / Rep-Global (all null expected)
p32s = []
for comp in [rep3_rt, crep3_rt, global_rt]:
    common = nested_rt.index.intersection(comp.index)
    _, p, _ = wilcoxon_pair(nested_rt.loc[common].values, comp.loc[common].values)
    p32s.append(p)
stats[32] = {'New N': N1, 'New p': f'min {min(p32s):.4f}'}
print(f"Row 32 RT Rep-Nested vs Rep-3/CRep-3/Rep-Global: p_vals={[round(p,4) for p in p32s]}")

# Row 33: Rep-Nested vs Rep-Local
common = nested_rt.index.intersection(local_rt.index)
W33, p33, r33 = wilcoxon_pair(nested_rt.loc[common].values, local_rt.loc[common].values)
stats[33] = {'New N': N1, 'New statistic': W33, 'New p': round(p33, 4), 'New effect size': r33}
print(f"Row 33 RT Rep-Nested vs Rep-Local: W={W33}, p={p33:.4f}, r={r33}")

# ── Rows 34-39: Constituent error rate (retroactive interference) ──
def constituent_er_per_participant(data, seq_name, n_chunk):
    """Error rate on first n_chunk positions of the ordinal structure."""
    rows = []
    sub = data[data['seq_name'] == seq_name]
    first_chunk = sub['sequences_structure'].iloc[0][:n_chunk]
    for pid in sub['participant_ID'].unique():
        sub_p = sub[sub['participant_ID'] == pid]
        errors = [1 if list(row['comparable_temp'][:n_chunk]) != list(first_chunk) else 0
                  for _, row in sub_p.iterrows()]
        rows.append((pid, 100 * np.mean(errors)))
    return pd.DataFrame(rows, columns=['pid', 'er']).set_index('pid')['er']

rep3_chunk = constituent_er_per_participant(df1, 'Repetition-3', 3)
crep3_chunk = constituent_er_per_participant(df1, 'control Repetition-3', 3)
rep4_chunk  = constituent_er_per_participant(df1, 'Repetition-4', 4)
crep4_chunk = constituent_er_per_participant(df1, 'control Repetition-4', 4)

stats[34] = {'New N': N1, 'New value': round(rep3_chunk.mean(), 2),
             'New SEM': round(scipy_stats.sem(rep3_chunk), 2)}
stats[35] = {'New N': N1, 'New value': round(crep3_chunk.mean(), 2),
             'New SEM': round(scipy_stats.sem(crep3_chunk), 2)}
print(f"Row 34 constituent ER Rep-3 '123' = {stats[34]['New value']} ± {stats[34]['New SEM']}%")
print(f"Row 35 constituent ER CRep-3 '123' = {stats[35]['New value']} ± {stats[35]['New SEM']}%")

common = rep3_chunk.index.intersection(crep3_chunk.index)
W36, p36, r36 = wilcoxon_pair(rep3_chunk.loc[common].values, crep3_chunk.loc[common].values)
stats[36] = {'New N': N1, 'New statistic': W36, 'New p': round(p36, 6), 'New effect size': r36}
print(f"Row 36 Rep-3 vs CRep-3 constituent: W={W36}, p={p36:.6f}, r={r36}")

stats[37] = {'New N': N1, 'New value': round(rep4_chunk.mean(), 2),
             'New SEM': round(scipy_stats.sem(rep4_chunk), 2)}
stats[38] = {'New N': N1, 'New value': round(crep4_chunk.mean(), 2),
             'New SEM': round(scipy_stats.sem(crep4_chunk), 2)}
print(f"Row 37 constituent ER Rep-4 '1234' = {stats[37]['New value']} ± {stats[37]['New SEM']}%")
print(f"Row 38 constituent ER CRep-4 '1234' = {stats[38]['New value']} ± {stats[38]['New SEM']}%")

common = rep4_chunk.index.intersection(crep4_chunk.index)
W39, p39, r39 = wilcoxon_pair(rep4_chunk.loc[common].values, crep4_chunk.loc[common].values)
stats[39] = {'New N': N1, 'New statistic': W39, 'New p': round(p39, 6), 'New effect size': r39}
print(f"Row 39 Rep-4 vs CRep-4 constituent: W={W39}, p={p39:.6f}, r={r39}")

# ──────────────────────────────────────────────
# 4.  EXP 2 — PYTHON-COMPUTABLE STATISTICS
# ──────────────────────────────────────────────

# Row 44: Pearson r between LoT complexity and mean edit distance (per-sequence means)
seq_dl_means = df2.groupby('seq_name')['distance_dl'].mean()
seq_lot = {sn: complexities_post_fit_exp1.get(sn, np.nan) for sn in seq_dl_means.index}
lot_vals = [seq_lot[s] for s in seq_dl_means.index]
dl_vals  = seq_dl_means.values
mask = ~np.isnan(lot_vals)
r44, p44 = scipy_stats.pearsonr(np.array(lot_vals)[mask], dl_vals[mask])
stats[44] = {'New N': N2, 'New statistic': round(r44, 4)}
print(f"\nRow 44 Pearson r(LoT vs edit_dist): r={r44:.4f}, p={p44:.4e}")

# Rows 67-69: Error rates for specific Exp2 sequences
for row_id, sname in [(67, 'play 4 tokens'), (68, 'sub-programs 1'), (69, 'Mirror-NoRep')]:
    er = per_participant_error_rate(df2, sname)
    stats[row_id] = {'New N': N2,
                     'New value': round(er.mean(), 2),
                     'New SEM':  round(scipy_stats.sem(er), 2)}
    print(f"Row {row_id} ER [{sname}] = {stats[row_id]['New value']} ± {stats[row_id]['New SEM']}%")

# ──────────────────────────────────────────────
# 5.  PARTICIPANTS SHEET
# ──────────────────────────────────────────────
participants_new = {
    1: {'Final N (new)': N1},
    2: {'Final N (new)': N2},
    3: {'Final N (new)': 77},   # Exp3 unchanged
}
print(f"\nParticipants sheet: Exp1 N={N1}, Exp2 N={N2}")

# ──────────────────────────────────────────────
# 6.  PER-SEQUENCE SHEET
# ──────────────────────────────────────────────

# Mapping: Per-sequence sheet name → seq_name in data
SEQ_MAP_EXP1 = {
    'Rep-2':      ('Repetition-2',              df1),
    'Rep-3':      ('Repetition-3',              df1),
    'Rep-4':      ('Repetition-4',              df1),
    'Rep-Nested': ('Repetition-Nested',         df1),
    'Rep-Global': ('control NoLocal nested',    df1),
    'Rep-Local':  ('control NoGlobal nested',   df1),
    'CRep-2':     ('control Repetition-2',      df1),
    'CRep-3':     ('control Repetition-3',      df1),
    'CRep-4':     ('control Repetition-4',      df1),
}

SEQ_MAP_EXP2 = {
    'Mirror-Rep':              ('Mirror-Rep',               df2),
    'Mirror-NoRep':            ('Mirror-NoRep',             df2),
    'NamedSubprogram-1':       ('sub-programs 1',           df2),
    'NamedSubprogram-2':       ('sub-programs 2',           df2),
    'Print':                   ('play',                     df2),
    'Print-4':                 ('play 4 tokens',            df2),
    'Control Mirror-Rep':      ('control Mirror-Rep',       df2),
    'Control Mirror-NoRep':    ('control Mirror-NoRep',     df2),
    'Control NamedSubprogram-1': ('control sub-programs 1', df2),
    'Control NamedSubprogram-2': ('control sub-programs 2', df2),
    'Control Print':           ('control play',             df2),
    'Control Print-4':         ('control play 4 tokens',    df2),
}

per_seq_results = {}
for label, (sname, df) in {**SEQ_MAP_EXP1, **SEQ_MAP_EXP2}.items():
    sub = df[df['seq_name'] == sname]
    if len(sub) == 0:
        per_seq_results[label] = {'er_mean': np.nan, 'er_sem': np.nan,
                                  'dl_mean': np.nan, 'dl_sem': np.nan}
        continue
    # Per-participant means
    pp_er = sub.groupby('participant_ID').apply(
        lambda g: 100 * (g['performance'] != 'success').mean())
    pp_dl = sub.groupby('participant_ID')['distance_dl'].mean()
    per_seq_results[label] = {
        'er_mean': round(pp_er.mean(), 2),
        'er_sem':  round(scipy_stats.sem(pp_er), 2),
        'dl_mean': round(pp_dl.mean(), 2),
        'dl_sem':  round(scipy_stats.sem(pp_dl), 2),
    }
    print(f"Per-seq [{label:35s}] ER={per_seq_results[label]['er_mean']:.2f}±{per_seq_results[label]['er_sem']:.2f}%  DL={per_seq_results[label]['dl_mean']:.2f}±{per_seq_results[label]['dl_sem']:.2f}")

# ──────────────────────────────────────────────
# 7.  MODEL COMPARISON — COPY EXISTING AIC VALUES
#     (EXT_only_REP data unchanged; R script must be re-run
#      separately for updated AIC values)
# ──────────────────────────────────────────────

# aic_values_exp2_rep (from params.py) = AIC from R LMM on EXT_only_REP
# These match what the R script produces; no change since Exp2 data unchanged.
model_aics = dict(zip(name_complexities, aic_values_exp2_rep))
model_aics_base = {
    'LoT Complexity': aic_values_exp2_rep_corrected[0],       # 6150.16
    'Subjective complexity': aic_values_exp2_rep_corrected[1], # 6269.45
}
print("\nModel comparison AIC values (from params.py, Exp2 EXT_only_REP):")
for m, a in model_aics.items():
    print(f"  {m}: {a}")

# ──────────────────────────────────────────────
# 8.  WRITE RESULTS TO EXCEL
# ──────────────────────────────────────────────

xl_path = os.path.join(ROOT, 'Manuscript_statistics_extraction.xlsx')

from openpyxl import load_workbook
from openpyxl.styles import PatternFill

wb = load_workbook(xl_path)

# ── Statistics sheet ─────────────────────────

ws_stats = wb['Statistics']

# Build header → column index mapping from row 1
hdr = {ws_stats.cell(1, c).value: c for c in range(1, ws_stats.max_column + 1)}
print("\nStatistics sheet columns:", {k: v for k, v in hdr.items() if k is not None})

col_new_n   = hdr.get('New N')
col_new_stat = hdr.get('New statistic')
col_new_df  = hdr.get('New df')
col_new_p   = hdr.get('New p')
col_new_es  = hdr.get('New effect size')
col_new_val = hdr.get('New value')
col_new_sem = hdr.get('New SEM')
col_id      = hdr.get('ID')

# Row offset: data starts at row 2 (row 1 is header)
for row in ws_stats.iter_rows(min_row=2, max_row=ws_stats.max_row):
    row_id_cell = row[col_id - 1]
    if row_id_cell.value is None:
        continue
    try:
        rid = int(row_id_cell.value)
    except (ValueError, TypeError):
        continue
    if rid not in stats:
        continue
    s = stats[rid]
    def _write(col, val):
        if col and val is not None:
            row[col - 1].value = val
    _write(col_new_n,    s.get('New N'))
    _write(col_new_stat, s.get('New statistic'))
    _write(col_new_df,   s.get('New df'))
    _write(col_new_p,    s.get('New p'))
    _write(col_new_es,   s.get('New effect size'))
    _write(col_new_val,  s.get('New value'))
    _write(col_new_sem,  s.get('New SEM'))

# ── Participants sheet ──────────────────────

ws_part = wb['Participants']
part_hdr = {ws_part.cell(1, c).value: c for c in range(1, ws_part.max_column + 1)}
col_exp     = part_hdr.get('Exp')
col_final_n = part_hdr.get('Final N (new)')

for row in ws_part.iter_rows(min_row=2, max_row=ws_part.max_row):
    if row[col_exp - 1].value is None:
        continue
    try:
        exp_id = int(row[col_exp - 1].value)
    except (ValueError, TypeError):
        continue
    if exp_id in participants_new and col_final_n:
        row[col_final_n - 1].value = participants_new[exp_id]['Final N (new)']

# ── Per-sequence sheet ───────────────────────

ws_seq = wb['Per-sequence']
seq_hdr = {ws_seq.cell(1, c).value: c for c in range(1, ws_seq.max_column + 1)}
col_seq_name    = seq_hdr.get('Sequence')
col_er_new      = seq_hdr.get('Error rate (new)')
col_er_sem_new  = seq_hdr.get('Error rate SEM (new)')
col_dl_new      = seq_hdr.get('Mean edit distance (new)')
col_dl_sem_new  = seq_hdr.get('Edit distance SEM (new)')

for row in ws_seq.iter_rows(min_row=2, max_row=ws_seq.max_row):
    sname_cell = row[col_seq_name - 1].value if col_seq_name else None
    if sname_cell is None:
        continue
    sname_cell = str(sname_cell).strip()
    if sname_cell not in per_seq_results:
        continue
    res = per_seq_results[sname_cell]
    def _ws(col, val):
        if col and val is not None and not (isinstance(val, float) and np.isnan(val)):
            row[col - 1].value = val
    _ws(col_er_new,     res['er_mean'])
    _ws(col_er_sem_new, res['er_sem'])
    _ws(col_dl_new,     res['dl_mean'])
    _ws(col_dl_sem_new, res['dl_sem'])

# ── Model comparison sheet ──────────────────

ws_mc = wb['Model comparison']
mc_hdr = {ws_mc.cell(1, c).value: c for c in range(1, ws_mc.max_column + 1)}
print("Model comparison columns:", {k: v for k, v in mc_hdr.items() if k is not None})

col_mc_model   = mc_hdr.get('Model')
col_mc_exp     = mc_hdr.get('Exp')
col_aic_new    = mc_hdr.get('AIC (new)')
col_daic_new   = mc_hdr.get('Delta AIC vs LoT (new)')
col_rank_new   = mc_hdr.get('Rank (new)')

# Fill AIC values for Exp 2 (Fig 3c) from params.py (EXT_only_REP, unchanged)
# model_aics keys use the name_complexities list
mc_model_name_map = {
    'LoT complexity':              'LoT Complexity',
    'Subjective complexity':       'Subjective Complexity',
    'Lempel-Ziv':                  'Lempel-Ziv',
    'zlib compression':            'Lempel-Ziv',   # possibly same
    'Shannon entropy (bigram)':    'Shannon Entropy Bigram',
    'Chunk complexity, global':    'Chunk Complexity Global',
    'Change complexity':           'Change Complexity',
    'Chunk complexity, local':     'Chunk Complexity Local',
    'Subsymmetries':               'Subsymetries',
}

aic_base = model_aics.get('LoT Complexity', np.nan)

for row in ws_mc.iter_rows(min_row=2, max_row=ws_mc.max_row):
    if col_mc_exp is None or col_mc_model is None:
        break
    exp_val = row[col_mc_exp - 1].value
    model_val = row[col_mc_model - 1].value
    if exp_val is None or model_val is None:
        continue

    model_str = str(model_val).strip()
    # Look up AIC from our dictionary
    name_key = mc_model_name_map.get(model_str.lower(),
               mc_model_name_map.get(model_str))

    # Try direct match in name_complexities
    aic_val = model_aics.get(model_str)
    if aic_val is None:
        for nc_name in name_complexities:
            if nc_name.lower() in model_str.lower() or model_str.lower() in nc_name.lower():
                aic_val = model_aics.get(nc_name)
                break

    if aic_val is not None and col_aic_new:
        row[col_aic_new - 1].value = round(aic_val, 2)
        if col_daic_new:
            row[col_daic_new - 1].value = round(aic_val - aic_base, 2)

# Exp 3 (rating vs model): LoT AIC = 6150.16, Subj = 6269.45
for row in ws_mc.iter_rows(min_row=2, max_row=ws_mc.max_row):
    if col_mc_exp is None:
        break
    exp_val = row[col_mc_exp - 1].value
    model_val = row[col_mc_model - 1].value if col_mc_model else None
    if exp_val == 3 and model_val:
        mstr = str(model_val).strip().lower()
        if 'lot' in mstr or 'minimal description' in mstr:
            if col_aic_new:
                row[col_aic_new - 1].value = round(aic_values_exp2_rep_corrected[0], 2)
            if col_daic_new:
                row[col_daic_new - 1].value = 0.0
        elif 'subjective' in mstr:
            if col_aic_new:
                row[col_aic_new - 1].value = round(aic_values_exp2_rep_corrected[1], 2)
            if col_daic_new:
                row[col_daic_new - 1].value = round(
                    aic_values_exp2_rep_corrected[1] - aic_values_exp2_rep_corrected[0], 2)

wb.save(xl_path)
print(f"\nSaved to {xl_path}")
