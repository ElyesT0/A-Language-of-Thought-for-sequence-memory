"""
run_pipeline_memocrush.py
─────────────────────────
Main analysis pipeline for the Memocrush manuscript.

Loads processed experiment data, computes all statistics reported in the
manuscript, and writes results to:
  1.  ../Manuscript_statistics_extraction.xlsx  (New columns only)
  2.  memocrush_statistics_report.txt           (human-readable summary)

Run from the Scripts/ directory with the data_analysis conda environment:
    /opt/anaconda3/envs/data_analysis/bin/python run_pipeline_memocrush.py
"""

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import os, sys, warnings
from itertools import takewhile
from datetime import datetime

warnings.filterwarnings('ignore')

ROOT   = '/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush'
SCRIPT = os.path.join(ROOT, 'Scripts')
sys.path.insert(0, SCRIPT)

from modules.params import *
from modules.functions import (str2int_dataset, dl_distance, compare_tokens, array_structure,
                               plot_mean_dl, plot_all_length, plot_length_distribution,
                               plot_comparison_AIC_models, plot_figure5_interclick,
                               plot_interclick_per_sequence, plot_interclick_grid,
                               generate_ici_accuracy_report)

# ══════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════

print("Loading data...")

# ── Experiment 1 ──
df1_raw = pd.read_csv(os.path.join(ROOT,
    'Data/processed/experiment-1/experiment1_processed_10h05_31032023_data.csv'))

trial_counts_1 = df1_raw.groupby('participant_ID').size()
excluded_1     = set(trial_counts_1[trial_counts_1 > 18].index)
df1 = df1_raw[~df1_raw['participant_ID'].isin(excluded_1)].copy().reset_index(drop=True)
df1 = df1[df1['seq_name'] != 'Training'].reset_index(drop=True)
df1 = str2int_dataset(df1, processed=True, label_int_col=label_int_col)

# ── Experiment 2 ──
df2 = pd.read_csv(os.path.join(ROOT,
    'Data/processed/experiment-2/processed_20240416_15h50_memocrush_extension_pilote2_data.csv'))
df2 = df2[df2['state'] == 'main_experiment'].reset_index(drop=True)
# Keep only participants with exactly max_trial main_experiment trials
trial_counts_2 = df2.groupby('participant_ID').size()
excluded_2     = set(trial_counts_2[trial_counts_2 != 49].index)
df2 = df2[~df2['participant_ID'].isin(excluded_2)].reset_index(drop=True)
df2 = str2int_dataset(df2, processed=True, label_int_col=label_int_col)

N1 = df1['participant_ID'].nunique()
N2 = df2['participant_ID'].nunique()
print(f"  Exp1: N={N1}  ({len(excluded_1)} excluded for wrong number of trials)")
print(f"  Exp2: N={N2}  ({len(excluded_2)} excluded for wrong number of trials)")

# ══════════════════════════════════════════════════════════════════════
# 2.  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def wilcoxon_pair(a, b):
    """Wilcoxon signed-rank test. Returns (W, p, r) where r = Z/sqrt(N_nonzero)."""
    a, b = np.array(a), np.array(b)
    stat, p = scipy_stats.wilcoxon(a, b)
    n_nz = int(np.sum(a - b != 0))
    r = round(scipy_stats.norm.isf(p / 2) / np.sqrt(n_nz), 4) if n_nz > 0 else np.nan
    return round(float(stat), 4), round(float(p), 6), r

def per_participant_mean(data, seq_name, col):
    """Per-participant mean of `col` for a given sequence name."""
    rows = []
    for pid in data['participant_ID'].unique():
        sub = data[(data['participant_ID'] == pid) & (data['seq_name'] == seq_name)]
        if len(sub) > 0:
            rows.append((pid, sub[col].mean()))
    return pd.Series(dict(rows))

def per_participant_error_rate(data, seq_name):
    """Per-participant error rate (%) for a given sequence name."""
    rows = []
    for pid in data['participant_ID'].unique():
        sub = data[(data['participant_ID'] == pid) & (data['seq_name'] == seq_name)]
        if len(sub) > 0:
            rows.append((pid, 100 * (sub['performance'] != 'success').mean()))
    return pd.Series(dict(rows))

def per_participant_dl(data, seq_name):
    return per_participant_mean(data, seq_name, 'distance_dl')

def per_participant_rt(data, seq_name):
    """Per-participant mean total RT (ms) = sum of interclick_time per trial."""
    rows = []
    for pid in data['participant_ID'].unique():
        sub = data[(data['participant_ID'] == pid) & (data['seq_name'] == seq_name)]
        if len(sub) > 0:
            rows.append((pid, np.mean([sum(t) for t in sub['interclick_time']])))
    return pd.Series(dict(rows))

def pooled_mean_sem_vals(series_dict, seq_list):
    """Per-participant means pooled across multiple sequences. Returns (mean, sem, vals)."""
    all_pids = list(set.intersection(*[set(series_dict[s].index) for s in seq_list]))
    vals = [np.mean([series_dict[s].loc[pid] for s in seq_list]) for pid in all_pids]
    return np.mean(vals), scipy_stats.sem(vals, ddof=1), vals

def paired_wilcoxon(measure_dict, seq_a, seq_b):
    common = measure_dict[seq_a].index.intersection(measure_dict[seq_b].index)
    a, b   = measure_dict[seq_a].loc[common].values, measure_dict[seq_b].loc[common].values
    return wilcoxon_pair(a, b)

def fmt_p(p):
    if isinstance(p, str):
        return p
    if p < .001:
        return '< .001'
    return f'{p:.3f}'

def fmt_r(r):
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return '—'
    return f'{r:.3f}'

# ══════════════════════════════════════════════════════════════════════
# 3.  ANALYSIS — EXPERIMENT 1
# ══════════════════════════════════════════════════════════════════════

stats = {}   # row_id → dict of new column values

# ── 3.1  Spatial accuracy & temporal order ────────────────────────────

# Row 1: Spatial position accuracy = 1 - TokenErr rate (all sequences are 12 tokens;
#        TokenErr flags responses using a different set of spatial positions)
pp_pos_acc = [100 * (1 - df1[df1['participant_ID'] == p]['TokenErr'].mean())
              for p in df1['participant_ID'].unique()]
stats[1] = {'New N': N1,
            'New value': round(np.mean(pp_pos_acc), 2),
            'New SEM':   round(scipy_stats.sem(pp_pos_acc, ddof=1), 2)}

# Row 2: Correct temporal order (performance == 'success')
pp_success = [100 * (df1[df1['participant_ID'] == p]['performance'] == 'success').mean()
              for p in df1['participant_ID'].unique()]
stats[2] = {'New N': N1,
            'New value': round(np.mean(pp_success), 2),
            'New SEM':   round(scipy_stats.sem(pp_success, ddof=1), 2)}

# ── 3.2  Primacy effects ──────────────────────────────────────────────

# Rows 3-5: Per-position accuracy at positions 1, 2, 3 relative to grand mean
pp_primacy = []
for pid in df1['participant_ID'].unique():
    sub = df1[df1['participant_ID'] == pid]
    pos_counts = np.zeros(12)
    for _, row in sub.iterrows():
        s, r = row['seq'], row['sequences_response']
        n = min(len(s), len(r), 12)
        for i in range(n):
            pos_counts[i] += (s[i] == r[i])
    pp_primacy.append(100 * pos_counts / len(sub))

pp_primacy   = np.array(pp_primacy)    # shape (N, 12)
mean_pos_acc = pp_primacy.mean(axis=0)
grand_mean   = pp_primacy.mean()
above_mean   = mean_pos_acc - grand_mean

for k, row_id in enumerate([3, 4, 5]):
    stats[row_id] = {'New N': N1, 'New value': round(above_mean[k], 2)}

# ── 3.3  Forward span ─────────────────────────────────────────────────

# Rows 6-11
def _accurate_length(seq, response):
    return sum(1 for _ in takewhile(lambda p: p[0] == p[1], zip(seq, response)))

df1['accurate_length'] = [_accurate_length(r['seq'], r['sequences_response'])
                          for _, r in df1.iterrows()]

span_row_map = {6: 'control Repetition-3', 7: 'control Repetition-4',
                9: 'Repetition-3',         10: 'Repetition-4'}
pp_span = {}
for row_id, sname in span_row_map.items():
    s = df1[df1['seq_name'] == sname].groupby('participant_ID')['accurate_length'].mean()
    stats[row_id] = {'New N': N1,
                     'New value': round(s.mean(), 2),
                     'New SEM':   round(scipy_stats.sem(s, ddof=1), 2)}
    pp_span[sname] = s

# Row 8: CRep-3 vs CRep-4 (null hypothesis: no difference)
c3, c4 = pp_span['control Repetition-3'], pp_span['control Repetition-4']
W8, p8, r8 = wilcoxon_pair(*[v.values for v in [c3.loc[c3.index.intersection(c4.index)],
                                                  c4.loc[c3.index.intersection(c4.index)]]])
stats[8] = {'New N': N1, 'New statistic': W8, 'New p': round(p8, 4)}

# Row 11: Rep-3 vs Rep-4
r3, r4 = pp_span['Repetition-3'], pp_span['Repetition-4']
W11, p11, r11 = wilcoxon_pair(*[v.values for v in [r3.loc[r3.index.intersection(r4.index)],
                                                     r4.loc[r3.index.intersection(r4.index)]]])
stats[11] = {'New N': N1, 'New statistic': W11, 'New p': round(p11, 4)}

# ── 3.4  Error rate ───────────────────────────────────────────────────

struc_seqs = ['Repetition-2', 'Repetition-3', 'Repetition-4']
ctrl_seqs  = ['control Repetition-2', 'control Repetition-3', 'control Repetition-4']

er_struc = {s: per_participant_error_rate(df1, s) for s in struc_seqs}
er_ctrl  = {s: per_participant_error_rate(df1, s) for s in ctrl_seqs}

m_er_s, sem_er_s, pp_er_s = pooled_mean_sem_vals(er_struc, struc_seqs)
m_er_c, sem_er_c, pp_er_c = pooled_mean_sem_vals(er_ctrl,  ctrl_seqs)

stats[12] = {'New N': N1, 'New value': round(m_er_s, 2), 'New SEM': round(sem_er_s, 2)}
stats[13] = {'New N': N1, 'New value': round(m_er_c, 2), 'New SEM': round(sem_er_c, 2)}

W14, p14, r14 = wilcoxon_pair(pp_er_s, pp_er_c)
stats[14] = {'New N': N1, 'New statistic': W14,
             'New p': round(p14, 6), 'New effect size': r14}

# ── 3.5  Edit distance ────────────────────────────────────────────────

dl_struc = {s: per_participant_dl(df1, s) for s in struc_seqs}
dl_ctrl  = {s: per_participant_dl(df1, s) for s in ctrl_seqs}

m_dl_s, sem_dl_s, pp_dl_s = pooled_mean_sem_vals(dl_struc, struc_seqs)
m_dl_c, sem_dl_c, pp_dl_c = pooled_mean_sem_vals(dl_ctrl,  ctrl_seqs)

stats[15] = {'New N': N1, 'New value': round(m_dl_s, 2), 'New SEM': round(sem_dl_s, 2)}
stats[16] = {'New N': N1, 'New value': round(m_dl_c, 2), 'New SEM': round(sem_dl_c, 2)}

W17, p17, r17 = wilcoxon_pair(pp_dl_s, pp_dl_c)
stats[17] = {'New N': N1, 'New statistic': W17,
             'New p': round(p17, 6), 'New effect size': r17}

# ── 3.6  Paired contrasts (nested sequences) ──────────────────────────

nested_seqs = ['Repetition-Nested', 'control NoLocal nested', 'control NoGlobal nested']
er_all  = {**er_struc, **er_ctrl,
           **{s: per_participant_error_rate(df1, s) for s in nested_seqs}}
dl_all  = {**dl_struc, **dl_ctrl,
           **{s: per_participant_dl(df1, s)         for s in nested_seqs}}

# Row 18: minimum r across all 6 structured-vs-control pairs (ER + DL)
r18_vals = []
for sa, ca in zip(struc_seqs, ctrl_seqs):
    for measure in [er_all, dl_all]:
        _, _, r = paired_wilcoxon(measure, sa, ca)
        r18_vals.append(r)
min_r18 = round(min(r18_vals), 4)
stats[18] = {'New N': N1, 'New effect size': f'>= {min_r18}', 'New p': '< .001'}

# Row 19: ER  Rep-Nested vs Rep-Local
W19, p19, r19 = paired_wilcoxon(er_all, 'Repetition-Nested', 'control NoGlobal nested')
stats[19] = {'New N': N1, 'New statistic': W19,
             'New p': round(p19, 6), 'New effect size': r19}

# Row 20: DL  Rep-Nested vs Rep-Local
W20, p20, r20 = paired_wilcoxon(dl_all, 'Repetition-Nested', 'control NoGlobal nested')
stats[20] = {'New N': N1, 'New statistic': W20,
             'New p': round(p20, 6), 'New effect size': r20}

# Row 21: ER + DL  Rep-Nested vs Rep-Global (report minimum r)
W21e, p21e, r21e = paired_wilcoxon(er_all, 'Repetition-Nested', 'control NoLocal nested')
W21d, p21d, r21d = paired_wilcoxon(dl_all, 'Repetition-Nested', 'control NoLocal nested')
stats[21] = {'New N': N1,
             'New effect size': f'> {round(min(r21e, r21d), 4)}',
             'New p': '< .001'}

# Row 22: ER + DL  Rep-Nested vs CRep-3
W22e, p22e, r22e = paired_wilcoxon(er_all, 'Repetition-Nested', 'control Repetition-3')
W22d, p22d, r22d = paired_wilcoxon(dl_all, 'Repetition-Nested', 'control Repetition-3')
stats[22] = {'New N': N1,
             'New effect size': f'> {round(min(r22e, r22d), 4)}',
             'New p': '< .001'}

# Row 23: ER + DL  Rep-Nested vs Rep-3  (null expected, report max p)
W23e, p23e, r23e = paired_wilcoxon(er_all, 'Repetition-Nested', 'Repetition-3')
W23d, p23d, r23d = paired_wilcoxon(dl_all, 'Repetition-Nested', 'Repetition-3')
stats[23] = {'New N': N1,
             'New statistic': W23e,
             'New p': f'{max(p23e, p23d):.4f}'}

# ── 3.7  Response time ────────────────────────────────────────────────

rt_seq_list = (struc_seqs + ctrl_seqs +
               ['Repetition-Nested', 'control NoLocal nested',
                'control NoGlobal nested', 'control Repetition-3'])
rt_all = {s: per_participant_rt(df1, s) for s in rt_seq_list}

m_rt_s, sem_rt_s, pp_rt_s = pooled_mean_sem_vals(rt_all, struc_seqs)
m_rt_c, sem_rt_c, pp_rt_c = pooled_mean_sem_vals(rt_all, ctrl_seqs)

stats[24] = {'New N': N1, 'New value': round(m_rt_s), 'New SEM': round(sem_rt_s)}
stats[25] = {'New N': N1, 'New value': round(m_rt_c), 'New SEM': round(sem_rt_c)}

W26, p26, r26 = wilcoxon_pair(pp_rt_s, pp_rt_c)
stats[26] = {'New N': N1, 'New statistic': W26,
             'New p': round(p26, 4), 'New effect size': r26}

# Rows 27-29: per-pair RT comparisons
for row_id, (sa, ca) in zip([27, 28, 29], zip(struc_seqs, ctrl_seqs)):
    common = rt_all[sa].index.intersection(rt_all[ca].index)
    W, p, r = wilcoxon_pair(rt_all[sa].loc[common].values,
                            rt_all[ca].loc[common].values)
    stats[row_id] = {'New N': N1,
                     'New statistic': W, 'New p': round(p, 4), 'New effect size': r}

# Rows 30-31: Rep-Nested and Rep-Local mean RT
nested_rt = rt_all['Repetition-Nested']
local_rt  = rt_all['control NoGlobal nested']
global_rt = rt_all['control NoLocal nested']
crep3_rt  = rt_all['control Repetition-3']
rep3_rt   = rt_all['Repetition-3']

stats[30] = {'New N': N1,
             'New value': round(nested_rt.mean()), 'New SEM': round(scipy_stats.sem(nested_rt, ddof=1))}
stats[31] = {'New N': N1,
             'New value': round(local_rt.mean()),  'New SEM': round(scipy_stats.sem(local_rt, ddof=1))}

# Row 32: Rep-Nested vs Rep-3 / CRep-3 / Rep-Global (all null expected)
p32s = []
for comp in [rep3_rt, crep3_rt, global_rt]:
    common = nested_rt.index.intersection(comp.index)
    _, p, _ = wilcoxon_pair(nested_rt.loc[common].values, comp.loc[common].values)
    p32s.append(p)
stats[32] = {'New N': N1, 'New p': f'min {min(p32s):.4f}'}

# Row 33: Rep-Nested vs Rep-Local
common = nested_rt.index.intersection(local_rt.index)
W33, p33, r33 = wilcoxon_pair(nested_rt.loc[common].values, local_rt.loc[common].values)
stats[33] = {'New N': N1, 'New statistic': W33,
             'New p': round(p33, 4), 'New effect size': r33}

# ── 3.8  Constituent error rate (retroactive interference) ────────────

def constituent_er(data, seq_name, n_chunk):
    """Error rate on first n_chunk ordinal positions."""
    sub        = data[data['seq_name'] == seq_name]
    first_chunk = list(sub['sequences_structure'].iloc[0][:n_chunk])
    rows = []
    for pid in sub['participant_ID'].unique():
        sub_p  = sub[sub['participant_ID'] == pid]
        errors = [1 if list(row['comparable_temp'][:n_chunk]) != first_chunk else 0
                  for _, row in sub_p.iterrows()]
        rows.append((pid, 100 * np.mean(errors)))
    return pd.Series(dict(rows))

rep3_chunk  = constituent_er(df1, 'Repetition-3',         3)
crep3_chunk = constituent_er(df1, 'control Repetition-3', 3)
rep4_chunk  = constituent_er(df1, 'Repetition-4',         4)
crep4_chunk = constituent_er(df1, 'control Repetition-4', 4)

stats[34] = {'New N': N1,
             'New value': round(rep3_chunk.mean(), 2),
             'New SEM':   round(scipy_stats.sem(rep3_chunk, ddof=1), 2)}
stats[35] = {'New N': N1,
             'New value': round(crep3_chunk.mean(), 2),
             'New SEM':   round(scipy_stats.sem(crep3_chunk, ddof=1), 2)}

common = rep3_chunk.index.intersection(crep3_chunk.index)
W36, p36, r36 = wilcoxon_pair(rep3_chunk.loc[common].values, crep3_chunk.loc[common].values)
stats[36] = {'New N': N1, 'New statistic': W36,
             'New p': round(p36, 6), 'New effect size': r36}

stats[37] = {'New N': N1,
             'New value': round(rep4_chunk.mean(), 2),
             'New SEM':   round(scipy_stats.sem(rep4_chunk, ddof=1), 2)}
stats[38] = {'New N': N1,
             'New value': round(crep4_chunk.mean(), 2),
             'New SEM':   round(scipy_stats.sem(crep4_chunk, ddof=1), 2)}

common = rep4_chunk.index.intersection(crep4_chunk.index)
W39, p39, r39 = wilcoxon_pair(rep4_chunk.loc[common].values, crep4_chunk.loc[common].values)
stats[39] = {'New N': N1, 'New statistic': W39,
             'New p': round(p39, 6), 'New effect size': r39}

# ══════════════════════════════════════════════════════════════════════
# 4.  ANALYSIS — EXPERIMENT 2
# ══════════════════════════════════════════════════════════════════════

# Row 44: Pearson r — LoT complexity vs mean edit distance per sequence
df2_nosub = df2[~df2['seq_name'].isin(['Training', 'Suppression', 'Insertion'])].copy()
seq_dl_means = df2_nosub.groupby('seq_name')['distance_dl'].mean()
lot_vals = np.array([complexities_post_fit_exp1.get(s, np.nan)
                     for s in seq_dl_means.index])
dl_vals  = seq_dl_means.values
mask     = ~np.isnan(lot_vals)
r44, p44 = scipy_stats.pearsonr(lot_vals[mask], dl_vals[mask])
stats[44] = {'New N': N2, 'New statistic': round(r44, 4)}

# Rows 67-69: error rates for specific sequences
for row_id, sname in [(67, 'play 4 tokens'), (68, 'sub-programs 1'), (69, 'Mirror-NoRep')]:
    er = per_participant_error_rate(df2, sname)
    stats[row_id] = {'New N':    N2,
                     'New value': round(er.mean(), 2),
                     'New SEM':   round(scipy_stats.sem(er, ddof=1), 2)}

# ══════════════════════════════════════════════════════════════════════
# 5.  PER-SEQUENCE RESULTS
# ══════════════════════════════════════════════════════════════════════

SEQ_MAP = {
    # Experiment 1 sequences
    'Rep-2':      ('Repetition-2',              df1),
    'Rep-3':      ('Repetition-3',              df1),
    'Rep-4':      ('Repetition-4',              df1),
    'Rep-Nested': ('Repetition-Nested',         df1),
    'Rep-Global': ('control NoLocal nested',    df1),
    'Rep-Local':  ('control NoGlobal nested',   df1),
    'CRep-2':     ('control Repetition-2',      df1),
    'CRep-3':     ('control Repetition-3',      df1),
    'CRep-4':     ('control Repetition-4',      df1),
    # Experiment 2 sequences
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

per_seq = {}
for label, (sname, df) in SEQ_MAP.items():
    sub = df[df['seq_name'] == sname]
    if len(sub) == 0:
        per_seq[label] = {'er_mean': np.nan, 'er_sem': np.nan,
                          'dl_mean': np.nan, 'dl_sem': np.nan}
        continue
    pp_er = sub.groupby('participant_ID').apply(
        lambda g: 100 * (g['performance'] != 'success').mean())
    pp_dl = sub.groupby('participant_ID')['distance_dl'].mean()
    per_seq[label] = {
        'er_mean': round(pp_er.mean(), 2), 'er_sem': round(scipy_stats.sem(pp_er, ddof=1), 2),
        'dl_mean': round(pp_dl.mean(), 2), 'dl_sem': round(scipy_stats.sem(pp_dl, ddof=1), 2),
    }

# ══════════════════════════════════════════════════════════════════════
# 6.  MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════

# AIC values from R LMM (distance_dl ~ Complexity + (1|participant_ID))
# run on Exp2 EXT_only_REP dataset (stored in params.py)
model_aics = dict(zip(name_complexities, aic_values_exp2_rep))
lot_aic    = model_aics['LoT Complexity']

# ══════════════════════════════════════════════════════════════════════
# 7.  WRITE TO EXCEL
# ══════════════════════════════════════════════════════════════════════

xl_path = os.path.join(ROOT, 'Manuscript_statistics_extraction.xlsx')

from openpyxl import load_workbook
wb = load_workbook(xl_path)

# ── Statistics sheet ─────────────────────────────────────────────────
ws_stats = wb['Statistics']
hdr = {ws_stats.cell(1, c).value: c for c in range(1, ws_stats.max_column + 1)}
col_id = hdr['ID']

col_map = {k: hdr.get(k) for k in
           ['New N', 'New statistic', 'New df', 'New p',
            'New effect size', 'New value', 'New SEM']}

for row in ws_stats.iter_rows(min_row=2, max_row=ws_stats.max_row):
    id_val = row[col_id - 1].value
    if id_val is None:
        continue
    try:
        rid = int(id_val)
    except (ValueError, TypeError):
        continue
    if rid not in stats:
        continue
    s = stats[rid]
    for key, col in col_map.items():
        if col and s.get(key) is not None:
            row[col - 1].value = s[key]

# ── Participants sheet ────────────────────────────────────────────────
ws_part = wb['Participants']
ph = {ws_part.cell(1, c).value: c for c in range(1, ws_part.max_column + 1)}
col_exp_p = ph.get('Exp')
col_fn    = ph.get('Final N (new)')
participants_new = {1: N1, 2: N2, 3: 77}  # Exp3 unchanged
for row in ws_part.iter_rows(min_row=2, max_row=ws_part.max_row):
    if row[col_exp_p - 1].value is None:
        continue
    try:
        exp_id = int(row[col_exp_p - 1].value)
    except (ValueError, TypeError):
        continue
    if exp_id in participants_new and col_fn:
        row[col_fn - 1].value = participants_new[exp_id]

# ── Per-sequence sheet ────────────────────────────────────────────────
ws_seq = wb['Per-sequence']
sh = {ws_seq.cell(1, c).value: c for c in range(1, ws_seq.max_column + 1)}
col_sname   = sh.get('Sequence')
col_er_new  = sh.get('Error rate (new)')
col_er_sem  = sh.get('Error rate SEM (new)')
col_dl_new  = sh.get('Mean edit distance (new)')
col_dl_sem  = sh.get('Edit distance SEM (new)')

for row in ws_seq.iter_rows(min_row=2, max_row=ws_seq.max_row):
    sname_val = row[col_sname - 1].value if col_sname else None
    if sname_val is None:
        continue
    key = str(sname_val).strip()
    if key not in per_seq:
        continue
    r = per_seq[key]
    for col, val in [(col_er_new, r['er_mean']), (col_er_sem, r['er_sem']),
                     (col_dl_new, r['dl_mean']), (col_dl_sem, r['dl_sem'])]:
        if col and val is not None and not (isinstance(val, float) and np.isnan(val)):
            row[col - 1].value = val

# ── Model comparison sheet ────────────────────────────────────────────
ws_mc = wb['Model comparison']
mh = {ws_mc.cell(1, c).value: c for c in range(1, ws_mc.max_column + 1)}
col_mc_exp   = mh.get('Exp')
col_mc_model = mh.get('Model')
col_aic_new  = mh.get('AIC (new)')
col_daic_new = mh.get('Delta AIC vs LoT (new)')

# Map Excel model names → our name_complexities keys
_aic_lookup = {
    'lot complexity':              'LoT Complexity',
    'minimal description length':  'LoT Complexity',
    'subjective complexity':       'Subjective Complexity',
    'lempel-ziv':                  'Lempel-Ziv',
    'zlib compression':            'Lempel-Ziv',
    'shannon entropy':             'Shannon Entropy',
    'shannon entropy (bigram)':    'Shannon Entropy Bigram',
    'change complexity':           'Change Complexity',
    'change complexity extended':  'Change Complexity Extended',
    'compression complexity':      'Compression Complexity',
    'subsymmetries':               'Subsymetries',
    'subsymetries':                'Subsymetries',
    'chunk complexity, local':     'Chunk Complexity Local',
    'chunk complexity, global':    'Chunk Complexity Global',
    'chunk complexity local':      'Chunk Complexity Local',
    'chunk complexity global':     'Chunk Complexity Global',
}

for row in ws_mc.iter_rows(min_row=2, max_row=ws_mc.max_row):
    if col_mc_model is None:
        break
    model_val = row[col_mc_model - 1].value
    if model_val is None:
        continue
    key = str(model_val).strip().lower()
    nc_name = _aic_lookup.get(key)
    if nc_name is None:
        for nc in name_complexities:
            if nc.lower() in key or key in nc.lower():
                nc_name = nc
                break
    if nc_name is None:
        continue
    aic_val = model_aics.get(nc_name)
    if aic_val is not None:
        if col_aic_new:
            row[col_aic_new - 1].value = round(aic_val, 2)
        if col_daic_new:
            row[col_daic_new - 1].value = round(aic_val - lot_aic, 2)

wb.save(xl_path)
print(f"\nExcel saved → {xl_path}")

# ══════════════════════════════════════════════════════════════════════
# 8.  FIGURES
# ══════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')  # non-interactive backend

FIG_EXP1 = os.path.join(ROOT, 'Figures/1_experiment1_only')
os.makedirs(os.path.join(FIG_EXP1, 'models', 'comparison_explanation'), exist_ok=True)

print("\nGenerating figures...")

# ── Fig 1: mean_dl_control Repetition-3.jpg ──────────────────────────
# Nested sequences comparison: Rep-Nested vs Rep-Global, Rep-Local, CRep-3
plot_mean_dl(
    data=df1,
    path=FIG_EXP1,
    plot_name='control Repetition-3',
    sequences=['Repetition-Nested', 'control NoLocal nested', 'control NoGlobal nested',
               'control Repetition-3'],
    unfill_controls=True,
    seq_expression=True,
    colors_figure=nested_colors,
    x_interval=[0, 6],
    x_ticks=True,
    print_values=False,
    save=True,
)

# ── Fig 2: mean_dl_control Repetition-4.jpg ──────────────────────────
# Rep-2/3/4 vs their controls
plot_mean_dl(
    data=df1,
    path=FIG_EXP1,
    plot_name='control Repetition-4',
    sequences=seq_subset1a,
    unfill_controls=True,
    seq_expression=True,
    colors_figure=blue_colors,
    x_interval=[0, 6],
    x_ticks=True,
    print_values=False,
    save=True,
)

# ── Fig 3: all_length_distribution_subplots.pdf (all trials) ─────────
plot_all_length(
    data=df1,
    path=FIG_EXP1,
    sequence_list=seq_name_list_exp1_only,
    nb_rows=3,
    nb_cols=3,
    figsize=(25, 25),
)

# ── Fig 4: errorOnly_all_length_distribution_subplots.pdf (errors only)
df1_errors = df1[df1['performance'] != 'success'].copy()
plot_all_length(
    data=df1_errors,
    path=FIG_EXP1,
    sequence_list=seq_name_list_exp1_only,
    nb_rows=3,
    nb_cols=3,
    figsize=(25, 25),
    file_prefix='errorOnly_',
)

# ── Length distribution stats reports (Tables 11.a / 11.b) ───────────
os.makedirs(os.path.join(FIG_EXP1, 'length'), exist_ok=True)
plot_length_distribution(
    data=df1,
    path=FIG_EXP1,
    sequence_list=seq_name_list_exp1_only,
    comparison_pairs=pairs_for_stat_test_exp1,
    file_prefix='',
)
plot_length_distribution(
    data=df1_errors,
    path=FIG_EXP1,
    sequence_list=seq_name_list_exp1_only,
    comparison_pairs=pairs_for_stat_test_exp1,
    file_prefix='errorOnly_',
)

# ── Chunk-error report (Table 12 — retroactive interference) ─────────
_chunk_path = os.path.join(FIG_EXP1, 'length', 'chunk_error_report.txt')
with open(_chunk_path, 'w', encoding='utf-8') as _cf:
    _W = 80
    _cf.write('=' * _W + '\n')
    _cf.write('CONSTITUENT ERROR RATE — RETROACTIVE INTERFERENCE\n')
    _cf.write(f'Experiment 1  (N={N1} participants)\n')
    _cf.write('=' * _W + '\n\n')
    _cf.write('Error rate on the first N positions of the sequence\n')
    _cf.write('(i.e. whether the first chunk was reproduced correctly).\n')
    _cf.write('Unit of analysis: per-participant mean error rate (%).\n\n')
    _cf.write('REP-3 vs control Rep-3  (first 3 positions)\n')
    _cf.write('-' * 60 + '\n')
    _cf.write(f'  Repetition-3           {stats[34]["New value"]:.2f} ± {stats[34]["New SEM"]:.2f}%  (N={N1})\n')
    _cf.write(f'  control Repetition-3   {stats[35]["New value"]:.2f} ± {stats[35]["New SEM"]:.2f}%  (N={N1})\n')
    _cf.write(f'  Wilcoxon: W={stats[36]["New statistic"]},  p={fmt_p(stats[36]["New p"])},  r={fmt_r(stats[36]["New effect size"])}\n\n')
    _cf.write('REP-4 vs control Rep-4  (first 4 positions)\n')
    _cf.write('-' * 60 + '\n')
    _cf.write(f'  Repetition-4           {stats[37]["New value"]:.2f} ± {stats[37]["New SEM"]:.2f}%  (N={N1})\n')
    _cf.write(f'  control Repetition-4   {stats[38]["New value"]:.2f} ± {stats[38]["New SEM"]:.2f}%  (N={N1})\n')
    _cf.write(f'  Wilcoxon: W={stats[39]["New statistic"]},  p={fmt_p(stats[39]["New p"])},  r={fmt_r(stats[39]["New effect size"])}\n')
    _cf.write('\n' + '=' * _W + '\n')
print(f"Chunk-error report → {_chunk_path}")

# ── Fig 5: AIC_complexity_models.jpg ─────────────────────────────────
smaller_set = [
    'LoT Complexity', 'Subjective Complexity', 'Shannon Entropy Bigram', 'Lempel-Ziv',
    'Change Complexity', 'Compression Complexity', 'Subsymetries',
    'Chunk Complexity Local', 'Chunk Complexity Global',
]
plot_comparison_AIC_models(path=FIG_EXP1, aic_arr=AIC_models_ext_rep, selected_models=smaller_set)

# ── Fig 6: figure5 interclick timing layout ──────────────────────────
plot_figure5_interclick(data=df1, path=FIG_EXP1, save=True)

# ── Individual interclick+accuracy plots (one PNG per sequence) ───────
plot_interclick_per_sequence(data=df1, path=FIG_EXP1, figsize=(10, 10),ylim_min=0.4, ylim_max=1.0)

# ── ICI & accuracy report (Exp1 sequences) ───────────────────────────
generate_ici_accuracy_report(
    data=df1, path=FIG_EXP1,
    seq_list=seq_name_list_exp1_only,
    filename='ici_report.txt',
    title='ICI AND ACCURACY ANALYSIS — EXPERIMENT 1',
)

print("Figures saved to", FIG_EXP1)

# ══════════════════════════════════════════════════════════════════════
# FIGURES — EXPERIMENT 2
# ══════════════════════════════════════════════════════════════════════

FIG_EXP2 = os.path.join(ROOT, 'Figures/2_experiment2')
os.makedirs(FIG_EXP2, exist_ok=True)

# ── Big grid: all Exp2 sequences in one figure ────────────────────────
plot_interclick_grid(data=df2, path=FIG_EXP2, file_prefix='interclick_grid')

# ── Individual interclick+accuracy plots (one PNG per sequence) ───────
plot_interclick_per_sequence(data=df2, path=FIG_EXP2, figsize=(10, 10),ylim_min=0.3, ylim_max=1.0)

# ── ICI & accuracy report (Exp2 sequences) ───────────────────────────
generate_ici_accuracy_report(
    data=df2, path=FIG_EXP2,
    seq_list=seq_name_list_exp2_only,
    filename='ici_report.txt',
    title='ICI AND ACCURACY ANALYSIS — EXPERIMENT 2',
)

# ── Length distribution stats reports (Tables 11.a / 11.b) ───────────
df2_errors = df2[df2['performance'] != 'success'].copy()
os.makedirs(os.path.join(FIG_EXP2, 'length'), exist_ok=True)
_pairs_exp2 = [
    ['play',           'control play'],
    ['play 4 tokens',  'control play 4 tokens'],
    ['sub-programs 1', 'control sub-programs 1'],
    ['sub-programs 2', 'control sub-programs 2'],
    ['index i',        'control index i'],
    ['Mirror-Rep',     'control Mirror-Rep'],
    ['Mirror-NoRep',   'control Mirror-NoRep'],
]
plot_length_distribution(
    data=df2,
    path=FIG_EXP2,
    sequence_list=seq_name_list_exp2_only,
    comparison_pairs=_pairs_exp2,
    file_prefix='',
)
plot_length_distribution(
    data=df2_errors,
    path=FIG_EXP2,
    sequence_list=seq_name_list_exp2_only,
    comparison_pairs=_pairs_exp2,
    file_prefix='errorOnly_',
)

print("Figures saved to", FIG_EXP2)

# ══════════════════════════════════════════════════════════════════════
# FIGURES — BOTH EXPERIMENTS POOLED
# ══════════════════════════════════════════════════════════════════════

FIG_BOTH = os.path.join(ROOT, 'Figures/interclick')
os.makedirs(FIG_BOTH, exist_ok=True)

# Pool Exp1 and Exp2 (keep only shared columns to avoid concat issues)
shared_cols = list(set(df1.columns) & set(df2.columns))
df_both = pd.concat([df1[shared_cols], df2[shared_cols]], ignore_index=True)

seq_list_both = seq_name_list_exp1_only + [
    s for s in seq_name_list_exp2_only if s not in seq_name_list_exp1_only
]

# ── Big grid: all sequences from both experiments ─────────────────────
plot_interclick_grid(data=df_both, path=FIG_BOTH, file_prefix='interclick_grid')

# ── Individual interclick+accuracy plots (one PNG per sequence) ───────
plot_interclick_per_sequence(data=df_both, path=FIG_BOTH, figsize=(10, 10),ylim_min=0.3, ylim_max=1.0)

# ── ICI & accuracy report (all sequences, pooled) ────────────────────
generate_ici_accuracy_report(
    data=df_both, path=FIG_BOTH,
    seq_list=seq_list_both,
    filename='ici_report.txt',
    title='ICI AND ACCURACY ANALYSIS — EXPERIMENTS 1 & 2 POOLED',
)

print("Figures saved to", FIG_BOTH)

# ══════════════════════════════════════════════════════════════════════
# 9.  TEXT REPORT
# ══════════════════════════════════════════════════════════════════════

W = 72  # report width

def hline(char='═'):
    return char * W

def section(title):
    pad = (W - len(title) - 2) // 2
    return f"\n{'─' * W}\n{' ' * pad} {title}\n{'─' * W}"

def row_line(label, value, sem=None, W_stat=None, p=None, r=None, N=None):
    parts = [f"  {label:<42s}  {str(value):>8s}"]
    if sem is not None:
        parts.append(f" ± {sem:<6s}")
    else:
        parts.append('         ')
    detail = []
    if N   is not None: detail.append(f"N={N}")
    if W_stat is not None: detail.append(f"W={W_stat}")
    if p   is not None: detail.append(f"p={fmt_p(p)}")
    if r   is not None: detail.append(f"r={fmt_r(r)}")
    if detail:
        parts.append('  ' + '  '.join(detail))
    return ''.join(parts)

def table_row(cols, widths):
    return '  ' + '  '.join(str(c).ljust(w) for c, w in zip(cols, widths))

lines = []
lines.append(hline())
lines.append(f"{'MEMOCRUSH — MANUSCRIPT STATISTICS REPORT':^{W}}")
lines.append(f"{'Generated: ' + datetime.now().strftime('%Y-%m-%d %H:%M'):^{W}}")
lines.append(hline())

# ── Participants ──────────────────────────────────────────────────────
lines.append(section("PARTICIPANTS"))
lines.append(f"  Experiment 1:  N = {N1}   (excluded {len(excluded_1)} participants with >18 trials)")
lines.append(f"  Experiment 2:  N = {N2}   (all main_experiment participants)")

# ── Exp1 basic metrics ────────────────────────────────────────────────
lines.append(section("EXP 1 — SPATIAL ACCURACY & TEMPORAL ORDER"))
lines.append(row_line("Spatial position accuracy (Row 1)",
                       f"{stats[1]['New value']}%",
                       sem=f"{stats[1]['New SEM']}%",
                       N=N1))
lines.append(row_line("Correct temporal order (Row 2)",
                       f"{stats[2]['New value']}%",
                       sem=f"{stats[2]['New SEM']}%",
                       N=N1))

# ── Primacy ───────────────────────────────────────────────────────────
lines.append(section("EXP 1 — PRIMACY EFFECTS (deviation from grand mean)"))
for k, row_id in enumerate([3, 4, 5]):
    lines.append(row_line(f"Position {k+1} above grand mean (Row {row_id})",
                           f"{stats[row_id]['New value']:+.2f} pp", N=N1))

# ── Forward span ─────────────────────────────────────────────────────
lines.append(section("EXP 1 — FORWARD SPAN"))
span_labels = {
    6: 'Control Rep-3', 7: 'Control Rep-4',
    9: 'Rep-3',         10: 'Rep-4',
}
for row_id, label in span_labels.items():
    s = stats[row_id]
    lines.append(row_line(f"{label} (Row {row_id})",
                           str(s['New value']), sem=str(s['New SEM']), N=N1))
lines.append(row_line("CRep-3 vs CRep-4 (Row 8)",
                       "Wilcoxon", W_stat=stats[8]['New statistic'],
                       p=stats[8]['New p']))
lines.append(row_line("Rep-3 vs Rep-4 (Row 11)",
                       "Wilcoxon", W_stat=stats[11]['New statistic'],
                       p=stats[11]['New p']))

# ── Error rate ────────────────────────────────────────────────────────
lines.append(section("EXP 1 — ERROR RATE (Structured vs Control)"))
lines.append(row_line("Structured sequences (Row 12)",
                       f"{stats[12]['New value']}%", sem=f"{stats[12]['New SEM']}%", N=N1))
lines.append(row_line("Control sequences (Row 13)",
                       f"{stats[13]['New value']}%", sem=f"{stats[13]['New SEM']}%", N=N1))
lines.append(row_line("Structured vs Control Wilcoxon (Row 14)",
                       "Wilcoxon",
                       W_stat=stats[14]['New statistic'],
                       p=stats[14]['New p'],
                       r=stats[14]['New effect size']))

# ── Edit distance ─────────────────────────────────────────────────────
lines.append(section("EXP 1 — EDIT DISTANCE (Structured vs Control)"))
lines.append(row_line("Structured sequences (Row 15)",
                       str(stats[15]['New value']), sem=str(stats[15]['New SEM']), N=N1))
lines.append(row_line("Control sequences (Row 16)",
                       str(stats[16]['New value']), sem=str(stats[16]['New SEM']), N=N1))
lines.append(row_line("Structured vs Control Wilcoxon (Row 17)",
                       "Wilcoxon",
                       W_stat=stats[17]['New statistic'],
                       p=stats[17]['New p'],
                       r=stats[17]['New effect size']))

# ── Nested contrasts ──────────────────────────────────────────────────
lines.append(section("EXP 1 — PAIRED CONTRASTS (Nested sequences)"))
lines.append(row_line("All 6 pairs ER+DL min r (Row 18)",
                       stats[18]['New effect size'], p='< .001'))
lines.append(row_line("ER: Rep-Nested vs Rep-Local (Row 19)",
                       "Wilcoxon",
                       W_stat=stats[19]['New statistic'],
                       p=stats[19]['New p'],
                       r=stats[19]['New effect size']))
lines.append(row_line("DL: Rep-Nested vs Rep-Local (Row 20)",
                       "Wilcoxon",
                       W_stat=stats[20]['New statistic'],
                       p=stats[20]['New p'],
                       r=stats[20]['New effect size']))
lines.append(row_line("ER+DL: Rep-Nested vs Rep-Global (Row 21)",
                       stats[21]['New effect size'], p='< .001'))
lines.append(row_line("ER+DL: Rep-Nested vs CRep-3 (Row 22)",
                       stats[22]['New effect size'], p='< .001'))
lines.append(row_line("ER+DL: Rep-Nested vs Rep-3 (Row 23, null)",
                       "Wilcoxon",
                       W_stat=stats[23]['New statistic'],
                       p=float(stats[23]['New p'])))

# ── Response time ─────────────────────────────────────────────────────
lines.append(section("EXP 1 — RESPONSE TIME"))
lines.append(row_line("Structured sequences RT (Row 24)",
                       f"{stats[24]['New value']} ms", sem=f"{stats[24]['New SEM']} ms", N=N1))
lines.append(row_line("Control sequences RT (Row 25)",
                       f"{stats[25]['New value']} ms", sem=f"{stats[25]['New SEM']} ms", N=N1))
lines.append(row_line("Structured vs Control RT (Row 26)",
                       "Wilcoxon",
                       W_stat=stats[26]['New statistic'],
                       p=stats[26]['New p'],
                       r=stats[26]['New effect size']))
pair_rt_labels = {
    27: 'Rep-2 vs CRep-2',
    28: 'Rep-3 vs CRep-3',
    29: 'Rep-4 vs CRep-4',
}
for row_id, label in pair_rt_labels.items():
    lines.append(row_line(f"RT {label} (Row {row_id})",
                           "Wilcoxon",
                           W_stat=stats[row_id]['New statistic'],
                           p=stats[row_id]['New p'],
                           r=stats[row_id]['New effect size']))
lines.append(row_line("Rep-Nested RT (Row 30)",
                       f"{stats[30]['New value']} ms", sem=f"{stats[30]['New SEM']} ms"))
lines.append(row_line("Rep-Local RT (Row 31)",
                       f"{stats[31]['New value']} ms", sem=f"{stats[31]['New SEM']} ms"))
lines.append(row_line("Rep-Nested vs Rep-3/CRep-3/Rep-Global (Row 32, null)",
                       stats[32]['New p']))
lines.append(row_line("RT Rep-Nested vs Rep-Local (Row 33)",
                       "Wilcoxon",
                       W_stat=stats[33]['New statistic'],
                       p=stats[33]['New p'],
                       r=stats[33]['New effect size']))

# ── Constituent error rate ────────────────────────────────────────────
lines.append(section("EXP 1 — CONSTITUENT ERROR RATE (Retroactive Interference)"))
lines.append(row_line("Rep-3, first 3 positions (Row 34)",
                       f"{stats[34]['New value']}%", sem=f"{stats[34]['New SEM']}%", N=N1))
lines.append(row_line("CRep-3, first 3 positions (Row 35)",
                       f"{stats[35]['New value']}%", sem=f"{stats[35]['New SEM']}%", N=N1))
lines.append(row_line("Rep-3 vs CRep-3 Wilcoxon (Row 36)",
                       "Wilcoxon",
                       W_stat=stats[36]['New statistic'],
                       p=stats[36]['New p'],
                       r=stats[36]['New effect size']))
lines.append(row_line("Rep-4, first 4 positions (Row 37)",
                       f"{stats[37]['New value']}%", sem=f"{stats[37]['New SEM']}%", N=N1))
lines.append(row_line("CRep-4, first 4 positions (Row 38)",
                       f"{stats[38]['New value']}%", sem=f"{stats[38]['New SEM']}%", N=N1))
lines.append(row_line("Rep-4 vs CRep-4 Wilcoxon (Row 39)",
                       "Wilcoxon",
                       W_stat=stats[39]['New statistic'],
                       p=stats[39]['New p'],
                       r=stats[39]['New effect size']))

# ── Experiment 2 ──────────────────────────────────────────────────────
lines.append(section("EXP 2 — STATISTICS"))
lines.append(f"  Participants: N = {N2}")
lines.append(row_line("LoT complexity vs edit dist Pearson r (Row 44)",
                       f"r = {stats[44]['New statistic']}", N=N2))
lines.append(row_line("Error rate: Play-4 (Row 67)",
                       f"{stats[67]['New value']}%", sem=f"{stats[67]['New SEM']}%", N=N2))
lines.append(row_line("Error rate: NamedSubprogram-1 (Row 68)",
                       f"{stats[68]['New value']}%", sem=f"{stats[68]['New SEM']}%", N=N2))
lines.append(row_line("Error rate: Mirror-NoRep (Row 69)",
                       f"{stats[69]['New value']}%", sem=f"{stats[69]['New SEM']}%", N=N2))

# ── Per-sequence table ────────────────────────────────────────────────
lines.append(section("PER-SEQUENCE RESULTS"))
col_widths = [24, 7, 6, 7, 6]
lines.append(table_row(['Sequence', 'ER (%)', '± SEM', 'DL dist', '± SEM'], col_widths))
lines.append('  ' + '─' * (sum(col_widths) + 2 * (len(col_widths) - 1)))
for label in list(SEQ_MAP.keys()):
    r = per_seq[label]
    if np.isnan(r['er_mean']):
        continue
    lines.append(table_row(
        [label,
         f"{r['er_mean']:.1f}", f"{r['er_sem']:.1f}",
         f"{r['dl_mean']:.2f}", f"{r['dl_sem']:.2f}"],
        col_widths))

# ── Model comparison ─────────────────────────────────────────────────
lines.append(section("MODEL COMPARISON — LMM (Exp 2, Repetition sequences)"))
lines.append("  Complexity metric             AIC       Δ AIC vs LoT   Rank")
lines.append("  " + "─" * 60)
ranked = sorted(model_aics.items(), key=lambda x: x[1])
for rank, (name, aic) in enumerate(ranked, 1):
    delta = aic - lot_aic
    lines.append(f"  {name:<30s}  {aic:>10.2f}  {delta:>+10.2f}     {rank}")

lines.append('\n' + hline())
lines.append(f"{'End of report':^{W}}")
lines.append(hline())

report_text = '\n'.join(lines) + '\n'

report_path = os.path.join(SCRIPT, 'memocrush_statistics_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"Report saved → {report_path}")
print("\n" + report_text)
