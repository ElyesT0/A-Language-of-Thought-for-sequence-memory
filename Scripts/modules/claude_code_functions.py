import os
import numpy as np
import pandas as pd
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "claude_reports")


def _ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check 1 — probe trial ratings per participant
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Dataset cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_subj_dataset(subj_dataset: pd.DataFrame):
    """
    Exclude bad participants and filter out probe/training trials.

    Exclusion criteria:
      - Responded > 2 on probe-easy more than 2 times.
      - Did not complete all sequences.

    Parameters
    ----------
    subj_dataset : raw DataFrame loaded from JSON.

    Returns
    -------
    subj_data_clean : DataFrame — no probe/training rows.
    subj_data_clean_with_probes : DataFrame — probes retained (for sanity checks).
    """
    # --- exclude bad participants ---
    mask_easy = "sequences_temp_tags == 'probe-easy'"
    df_filtered = subj_dataset.query(mask_easy)
    dict_flagged = {}
    flagged = []
    holder = []
    previous_p = 0

    for _, row in df_filtered.iterrows():
        response = row['participant_response']
        p = row['participant_prolific_id']
        if response > 2:
            if previous_p and previous_p != p:
                dict_flagged[previous_p] = holder
                holder = []
            flagged.append(p)
            holder.append(response)
            previous_p = p
    if holder:
        dict_flagged[previous_p] = holder

    flagged = np.unique(flagged)
    excluded = [k for k, v in dict_flagged.items() if len(v) > 2]

    df = subj_dataset.drop(subj_dataset[subj_dataset['participant_prolific_id'].isin(excluded)].index)

    # --- keep only participants who completed all sequences ---
    all_sequences = df['sequences_temp_tags'].unique()
    n_sequences = len(all_sequences)
    valid_participants = (
        df.groupby('participant_prolific_id')['sequences_temp_tags']
        .nunique()
        .pipe(lambda s: s[s == n_sequences].index)
    )
    df = df[df['participant_prolific_id'].isin(valid_participants)]

    all_durations = (
        df.groupby('participant_prolific_id')
        .apply(lambda g: (g.iloc[0].last_click - g.iloc[0].participant_startTime) / 60000,
               include_groups=False)
        .tolist()
    )
    mean_duration = np.mean(all_durations)
    print('Initial N :', len(subj_dataset.participant_prolific_id.unique()))
    print('Excluded  :', len(subj_dataset.participant_prolific_id.unique()) - len(df.participant_prolific_id.unique()))
    print('Final N   :', len(df.participant_prolific_id.unique()), 'participants')
    print('Mean duration :', round(mean_duration, 2), 'mins')

    subj_data_clean_with_probes = df.copy()

    mask = df['sequences_temp_tags'].str.contains('probe-easy|probe-hard|training')
    subj_data_clean = df[~mask].reset_index(drop=True)

    return subj_data_clean, subj_data_clean_with_probes


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

PROBE_TAGS = {"probe-easy", "probe-hard-1", "probe-hard-2", "probe-hard-3", "probe-hard-4", "probe-hard-5"}


def sanity_check_probe_ratings(subj_data_clean_with_probes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean ± SEM of complexity responses per participant for each probe
    sequence, and save the result as a text report.

    Parameters
    ----------
    subj_data_clean_with_probes : DataFrame
        Full cleaned dataset **before** probe rows were removed.
        Must contain columns: sequences_temp_tags, participant_prolific_id,
        participant_response.

    Returns
    -------
    summary : DataFrame
        MultiIndex (probe_sequence, participant_prolific_id) with mean and SEM.
    """
    probe_mask = subj_data_clean_with_probes["sequences_temp_tags"].isin(PROBE_TAGS)
    probe_df = subj_data_clean_with_probes[probe_mask].copy()

    # Mean per (probe, participant) — each participant may have repeated the probe
    per_pp = (
        probe_df
        .groupby(["sequences_temp_tags", "participant_prolific_id"])["participant_response"]
        .mean()
    )

    # Summary across participants: mean of means + SEM
    summary = (
        per_pp
        .groupby("sequences_temp_tags")
        .agg(
            mean=lambda x: round(x.mean(), 3),
            sem=lambda x: round(x.sem(), 3),
            N="count",
        )
    )

    # ── report ────────────────────────────────────────────────────────────────
    _ensure_reports_dir()
    date_str = datetime.now().strftime("%d-%m-%Y")
    report_path = os.path.join(REPORTS_DIR, f"{date_str}-sanity_check_probes.txt")

    lines = [
        "Sanity Check 1 — Probe Trial Complexity Ratings",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"N participants: {probe_df['participant_prolific_id'].nunique()}",
        "=" * 55,
        "",
        f"{'Probe sequence':<25} {'Mean':>8} {'SEM':>8} {'N':>6}",
        "-" * 55,
    ]
    for probe, row in summary.iterrows():
        lines.append(f"{probe:<25} {row['mean']:>8.3f} {row['sem']:>8.3f} {int(row['N']):>6}")
    lines += ["", "Expected pattern: probe-easy < probe-hard-*"]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved → {report_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check 2 — overall rating distribution (no probes)
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check_overall_ratings(subj_data_clean: pd.DataFrame) -> dict:
    """
    Compute mean, SD, and value distribution of subjective complexity ratings
    over all non-probe trials, and save the result as a text report.

    Parameters
    ----------
    subj_data_clean : DataFrame
        Cleaned dataset with probe rows already removed.
        Must contain column: participant_response.

    Returns
    -------
    stats_dict : dict with keys mean, sd, min, max, value_counts
    """
    responses = subj_data_clean["participant_response"].dropna()

    mean_val  = round(responses.mean(), 3)
    sd_val    = round(responses.std(), 3)
    min_val   = int(responses.min())
    max_val   = int(responses.max())
    val_counts = responses.value_counts().sort_index()
    val_pct    = (val_counts / len(responses) * 100).round(1)

    stats_dict = {
        "mean": mean_val,
        "sd": sd_val,
        "min": min_val,
        "max": max_val,
        "value_counts": val_counts,
    }

    # ── report ────────────────────────────────────────────────────────────────
    _ensure_reports_dir()
    date_str = datetime.now().strftime("%d-%m-%Y")
    report_path = os.path.join(REPORTS_DIR, f"{date_str}-sanity_check_overall_ratings.txt")

    lines = [
        "Sanity Check 2 — Overall Subjective Complexity Rating Distribution",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total responses: {len(responses)}",
        "=" * 50,
        "",
        f"  Mean : {mean_val}",
        f"  SD   : {sd_val}",
        f"  Range: {min_val} – {max_val}",
        "",
        "Response distribution:",
        f"  {'Value':>6}  {'Count':>7}  {'%':>6}",
        "  " + "-" * 24,
    ]
    for val, cnt in val_counts.items():
        lines.append(f"  {int(val):>6}  {cnt:>7}  {val_pct[val]:>5.1f}%")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved → {report_path}")
    return stats_dict


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "claude", "subjective")


def plot_distribution_subj(subj_data_clean: pd.DataFrame) -> None:
    """
    Plot the distribution of subjective complexity ratings over all trials
    and save to reports/claude/subjective/all_responses_distribution.pdf.

    Parameters
    ----------
    subj_data_clean : DataFrame
        Cleaned dataset with probe rows removed.
        Must contain column: participant_response.
    """
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "all_responses_distribution.png")

    responses = subj_data_clean["participant_response"].dropna()
    values = sorted(responses.unique().astype(int))
    counts = responses.value_counts().sort_index()
    pct = (counts / len(responses) * 100)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(values, [counts[v] for v in values], color="#800020", edgecolor="white", width=0.6)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + len(responses) * 0.005,
            f"{pct[v]:.1f}%",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_ylabel("Count", fontsize=11, rotation = 0, ha="left", va = "bottom")
    ax.yaxis.set_label_coords(-0.05,1.02)
    ax.set_xticks(values)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi = 350)
    plt.show()
    print(f"Plot saved → {save_path}")
