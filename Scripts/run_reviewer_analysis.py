"""
Reviewer analysis pipeline for the memocrush sequence reproduction experiment.
Run from the Scripts/ directory:

    /opt/anaconda3/envs/data_analysis/bin/python run_reviewer_analysis.py

Outputs are saved under:
    Figures/review_response/<analysis_type>/exp1/
    Figures/review_response/<analysis_type>/exp2/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from modules import *
from modules.geometry_effect import add_geom_columns, run_geometry_analysis

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = '/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush'

DATA_EXP1 = os.path.join(
    ROOT, 'Data/processed/experiment-1/experiment1_processed_10h05_31032023_data.csv'
)
DATA_EXP2 = os.path.join(
    ROOT, 'Data/processed/experiment-2/processed_20240416_15h50_memocrush_extension_pilote2_data.csv'
)

FIG_GEOM_EXP1 = os.path.join(ROOT, 'Figures/review_response/geometry_analysis/exp1')
FIG_GEOM_EXP2 = os.path.join(ROOT, 'Figures/review_response/geometry_analysis/exp2')

# ── Load data ──────────────────────────────────────────────────────────────────

print('Loading data...')
df1 = add_geom_columns(pd.read_csv(DATA_EXP1))
df2_raw = pd.read_csv(DATA_EXP2)
# Exclude training trials present in exp2 data
df2 = add_geom_columns(df2_raw[df2_raw['seq_name'] != 'Training'].copy())

print(f'  Exp1: {len(df1)} trials, {df1["participant_ID"].nunique()} participants')
print(f'  Exp2: {len(df2)} trials, {df2["participant_ID"].nunique()} participants')

# ── Geometry analysis ──────────────────────────────────────────────────────────

print('\n--- Geometry analysis: Experiment 1 ---')
run_geometry_analysis(
    df1,
    fig_dir=FIG_GEOM_EXP1,
    seq_names=seq_name_list_exp1_only,
    exp_label='Experiment 1 (base repetition sequences)',
)

print('\n--- Geometry analysis: Experiment 2 ---')
run_geometry_analysis(
    df2,
    fig_dir=FIG_GEOM_EXP2,
    seq_names=seq_name_list,
    exp_label='Experiment 2 (all sequences)',
)

print('\nDone.')
