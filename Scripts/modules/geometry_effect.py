import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kruskal, mannwhitneyu
from datetime import datetime

from modules.params import seq_name_list, bar_frame_width, title_size, padding_size
from modules.functions import _adaptive_dpi
from modules.stats import eliminate_outliers


# ---------------------------------------------------------------------------
# Geometry classification
# ---------------------------------------------------------------------------

def _circular_dist(a, b, n=6):
    d = abs(a - b)
    return min(d, n - d)


def _get_unique_ordered(seq):
    seen = []
    for x in seq:
        if x not in seen:
            seen.append(x)
    return seen


def _classify_geom(seq, n=6):
    """
    Classify the geometric pattern of a sequence on an n-vertex regular polygon.

    Computes the sorted tuple of cyclic min-circular-distances between consecutive
    unique positions (in order of first appearance) and matches to a named pattern.

    Returns one of: 'dist-1','dist-2','dist-3' (Rep-2),
    'rot-1','triangle','2groups' (Rep-3/Nested),
    'rot-1','zhang-23','zhang-30' (Rep-4/Mirror*/Play4), or 'other'.

    Validated at 100% accuracy against complexity_estimation experiment reference data
    for Rep-2/3/4, Nested, Mirror*, Play4 sequences.
    """
    unique = _get_unique_ordered(seq)
    k = len(unique)
    jumps = tuple(sorted(
        _circular_dist(unique[i], unique[(i + 1) % k], n)
        for i in range(k)
    ))
    if k == 2:
        mapping = {(1, 1): 'dist-1', (2, 2): 'dist-2', (3, 3): 'dist-3'}
    elif k == 3:
        mapping = {(1, 1, 2): 'rot-1', (2, 2, 2): 'triangle', (1, 2, 3): '2groups'}
    elif k == 4:
        mapping = {
            (1, 1, 1, 3): 'rot-1',
            (2, 2, 3, 3): 'zhang-23',
            (1, 1, 3, 3): 'zhang-30',
        }
    else:
        return 'other'
    return mapping.get(jumps, 'other')


def _parse_seq(seq_val):
    if isinstance(seq_val, list):
        return seq_val
    return ast.literal_eval(seq_val)


# ---------------------------------------------------------------------------
# Public: add geometry columns
# ---------------------------------------------------------------------------

def add_geom_columns(df):
    """
    Add 'starting_vertex' (int) and 'geom_tag' (str) columns from the 'seq' column.
    """
    df = df.copy()
    parsed = df['seq'].apply(_parse_seq)
    df['starting_vertex'] = parsed.apply(lambda s: s[0])
    df['geom_tag'] = parsed.apply(_classify_geom)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(msg, report=None):
    """Print to stdout and optionally write to an open report file."""
    print(msg)
    if report is not None:
        report.write(msg + '\n')


def _aggregate_per_participant(subset, dl=True):
    """Return dict {participant_ID: mean_DL_or_error_rate} for a given subset."""
    result = {}
    for pid, grp in subset.groupby('participant_ID'):
        if dl:
            result[pid] = float(np.mean(grp['distance_dl'].to_numpy()))
        else:
            n = len(grp)
            n_success = int((grp['performance'] == 'success').sum())
            result[pid] = 100.0 * (1 - n_success / n)
    return result


def _mwu_effect_size(u, n1, n2):
    """Effect size r for Mann-Whitney U: r = |Z| / sqrt(N)."""
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mean_u) / std_u
    return abs(z) / np.sqrt(n1 + n2)


def _pairwise_mwu(tag_agg, tag1, tag2):
    """Mann-Whitney U between two geom_tag groups (between-participants design)."""
    a = np.array(list(tag_agg[tag1].values()))
    b = np.array(list(tag_agg[tag2].values()))
    if len(a) < 3 or len(b) < 3:
        return None
    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    r = _mwu_effect_size(stat, len(a), len(b))
    return stat, p, r, len(a), len(b)


# ---------------------------------------------------------------------------
# Analysis 1: Effect of starting vertex
# ---------------------------------------------------------------------------

def test_starting_vertex_effect(df, seq_names=None, path=None, dl=True, report=None):
    """
    Kruskal-Wallis test of starting vertex (0-5) on performance, run separately
    for each sequence type. Between-participants design.

    Parameters
    ----------
    df : DataFrame with 'starting_vertex', 'seq_name', 'participant_ID',
         'distance_dl', 'performance' columns.
    seq_names : list of str, optional. Defaults to all sequences.
    path : str — directory to save the figure PDF. None = plt.show().
    dl : bool — if True use DL distance, else use error rate.
    report : open file object, optional — stats are written here in addition to stdout.
    """
    if seq_names is None:
        seq_names = seq_name_list

    metric_label = 'Mean DL distance' if dl else 'Mean error rate (%)'
    _log(f'\n{"="*60}', report)
    _log('ANALYSIS 1: Effect of starting vertex', report)
    _log(f'Metric: {metric_label}', report)
    _log(f'{"="*60}\n', report)

    plot_data = {}

    for seq in seq_names:
        sub = df[df['seq_name'] == seq]
        if sub.empty:
            continue

        groups = {}
        for v, grp in sub.groupby('starting_vertex'):
            agg = _aggregate_per_participant(grp, dl=dl)
            if len(agg) >= 2:
                groups[int(v)] = np.array(list(agg.values()))

        if len(groups) < 2:
            continue

        stat, p = kruskal(*groups.values())
        _log(f'[{seq}]', report)
        _log(f'  Kruskal-Wallis H = {round(stat, 3)}, p = {round(p, 4)}', report)
        for v, arr in sorted(groups.items()):
            _log(f'  vertex {v}: n={len(arr)}, mean={round(float(np.mean(arr)), 3)}, '
                 f'SEM={round(float(stats.sem(arr)), 3)}', report)

        plot_data[seq] = {v: (float(np.mean(arr)), float(stats.sem(arr)), len(arr))
                          for v, arr in groups.items()}

    if not plot_data:
        return

    all_vertices = sorted({v for d in plot_data.values() for v in d})
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(all_vertices)))
    vertex_color = {v: cmap[i] for i, v in enumerate(all_vertices)}

    plt.rcParams['figure.facecolor'] = 'white'
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_data) * 0.6)))

    for y_pos, seq in enumerate(plot_data):
        for vertex, (mean, sem, _) in sorted(plot_data[seq].items()):
            ax.errorbar(mean, y_pos + (vertex - 2.5) * 0.08,
                        xerr=sem, fmt='o', capsize=3,
                        linewidth=bar_frame_width * 0.5,
                        color=vertex_color[vertex],
                        label=f'vertex {vertex}' if y_pos == 0 else '')

    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(list(plot_data.keys()), fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(metric_label, fontsize=title_size, labelpad=padding_size)
    ax.set_title('Effect of starting vertex on performance', fontsize=title_size)
    handles = [plt.Line2D([0], [0], marker='o', color=vertex_color[v],
                           linestyle='', label=f'vertex {v}') for v in all_vertices]
    ax.legend(handles=handles, loc='lower right', fontsize=9, title='Starting vertex')
    plt.tight_layout()

    if path:
        fname = os.path.join(path, 'geometry_starting_vertex.pdf')
        plt.savefig(fname, bbox_inches='tight', dpi=_adaptive_dpi())
        _log(f'\nFigure saved: {fname}', report)
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Analysis 2: Effect of shape (crossing vs clean rotation)
# ---------------------------------------------------------------------------

_FAMILY_TAGS = {
    'Rep-2':  ['dist-1', 'dist-2', 'dist-3'],
    'Rep-3':  ['rot-1', 'triangle', '2groups'],
    'Rep-4':  ['rot-1', 'zhang-23', 'zhang-30'],
    'Nested': ['rot-1', 'triangle', '2groups'],
    'Mirror': ['rot-1', 'zhang-23', 'zhang-30'],
    'Play4':  ['rot-1', 'zhang-23', 'zhang-30'],
}

_SEQ_FAMILY = {
    'Repetition-2':            'Rep-2',
    'control Repetition-2':    'Rep-2',
    'Repetition-3':            'Rep-3',
    'control Repetition-3':    'Rep-3',
    'Repetition-4':            'Rep-4',
    'control Repetition-4':    'Rep-4',
    'Repetition-Nested':       'Nested',
    'control NoLocal nested':  'Nested',
    'control NoGlobal nested': 'Nested',
    'Mirror-Rep':              'Mirror',
    'control Mirror-Rep':      'Mirror',
    'Mirror-NoRep':            'Mirror',
    'control Mirror-NoRep':    'Mirror',
    'play 4 tokens':           'Play4',
    'control play 4 tokens':   'Play4',
}

_TAG_COLORS = {
    'rot-1':    'black',
    'dist-1':   'black',
    'triangle': 'royalblue',
    '2groups':  'forestgreen',
    'zhang-23': 'crimson',
    'zhang-30': 'darkorange',
    'dist-2':   'royalblue',
    'dist-3':   'forestgreen',
    'other':    'grey',
}


def test_shape_effect(df, seq_names=None, path=None, dl=True, report=None):
    """
    Between-participants Kruskal-Wallis test of geometric shape on performance,
    run separately for each sequence type. Pairwise Mann-Whitney U tests compare
    clean rotation (rot-1/dist-1) against each crossing variant.

    Parameters
    ----------
    df : DataFrame with 'geom_tag', 'seq_name', etc.
    seq_names : list of str, optional.
    path : str — directory to save the figure PDF.
    dl : bool.
    report : open file object, optional.
    """
    if seq_names is None:
        seq_names = list(_SEQ_FAMILY.keys())

    metric_label = 'Mean DL distance' if dl else 'Mean error rate (%)'
    _log(f'\n{"="*60}', report)
    _log('ANALYSIS 2: Effect of shape (crossing vs clean rotation)', report)
    _log(f'Metric: {metric_label}', report)
    _log(f'{"="*60}\n', report)

    plot_rows = []

    for seq in seq_names:
        sub = df[df['seq_name'] == seq]
        if sub.empty:
            continue

        family = _SEQ_FAMILY.get(seq)
        tags_order = _FAMILY_TAGS.get(family, []) if family else []

        tag_agg = {}
        for tag in tags_order:
            agg = _aggregate_per_participant(sub[sub['geom_tag'] == tag], dl=dl)
            if len(agg) >= 3:
                tag_agg[tag] = agg

        if len(tag_agg) < 2:
            continue

        _log(f'[{seq}]', report)
        for tag, agg in tag_agg.items():
            vals = list(agg.values())
            _log(f'  {tag}: n={len(vals)}, mean={round(np.mean(vals),3)}, '
                 f'SEM={round(float(stats.sem(vals)),3)}', report)
            plot_rows.append((seq, tag, float(np.mean(vals)), float(stats.sem(vals))))

        groups = [np.array(list(v.values())) for v in tag_agg.values()]
        if all(len(g) >= 3 for g in groups):
            stat, p = kruskal(*groups)
            _log(f'  Kruskal-Wallis H = {round(stat,3)}, p = {round(p,4)}', report)

        clean_tag = 'rot-1' if 'rot-1' in tag_agg else ('dist-1' if 'dist-1' in tag_agg else None)
        if clean_tag:
            for tag in tag_agg:
                if tag == clean_tag:
                    continue
                result = _pairwise_mwu(tag_agg, clean_tag, tag)
                if result:
                    stat_u, p_u, r_u, n1, n2 = result
                    _log(f'  Mann-Whitney {clean_tag} vs {tag}: U={round(stat_u,1)}, '
                         f'p={round(p_u,4)}, r={round(r_u,3)}, n=({n1},{n2})', report)
        _log('', report)

    if not plot_rows:
        return

    seq_order = list(dict.fromkeys(r[0] for r in plot_rows))
    plt.rcParams['figure.facecolor'] = 'white'
    fig, ax = plt.subplots(figsize=(8, max(3, len(seq_order) * 0.7)))

    shown_tags = []
    for y_pos, seq in enumerate(seq_order):
        rows = [(t, m, s) for sn, t, m, s in plot_rows if sn == seq]
        for tag, mean, sem in rows:
            color = _TAG_COLORS.get(tag, 'grey')
            ax.errorbar(mean, y_pos, xerr=sem, fmt='o', capsize=4,
                        linewidth=bar_frame_width * 0.6, color=color,
                        label=tag if tag not in shown_tags else '')
            if tag not in shown_tags:
                shown_tags.append(tag)

    ax.set_yticks(range(len(seq_order)))
    ax.set_yticklabels(seq_order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(metric_label, fontsize=title_size, labelpad=padding_size)
    ax.set_title('Effect of geometric shape on performance', fontsize=title_size)
    handles = [plt.Line2D([0], [0], marker='o', color=_TAG_COLORS.get(t, 'grey'),
                           linestyle='', label=t) for t in shown_tags]
    ax.legend(handles=handles, loc='lower right', fontsize=9, title='Geometry')
    plt.tight_layout()

    if path:
        fname = os.path.join(path, 'geometry_shape_effect.pdf')
        plt.savefig(fname, bbox_inches='tight', dpi=_adaptive_dpi())
        _log(f'Figure saved: {fname}', report)
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Analysis 3: Zhang et al. (2022) patterns for Rep-4 (+ control Rep-4)
# ---------------------------------------------------------------------------

_ZHANG_SEQ_NAMES = ['Repetition-4', 'control Repetition-4']
_ZHANG_TAGS = ['rot-1', 'zhang-23', 'zhang-30']
_ZHANG_COLORS = {'rot-1': 'black', 'zhang-23': 'crimson', 'zhang-30': 'darkorange'}
_ZHANG_JITTER = {'rot-1': -0.15, 'zhang-23': 0.0, 'zhang-30': 0.15}


def test_zhang_patterns(df, path=None, dl=True, report=None):
    """
    Between-participants comparison of rot-1, zhang-23, and zhang-30 geometric
    patterns for Repetition-4 and control Repetition-4.

    Uses Kruskal-Wallis for the 3-group test and pairwise Mann-Whitney U with
    effect size r. Trials with 'other' geom_tag are excluded.

    Parameters
    ----------
    df : DataFrame with 'geom_tag', 'seq_name', etc.
    path : str — directory to save the figure PDF.
    dl : bool.
    report : open file object, optional.
    """
    metric_label = 'Mean DL distance' if dl else 'Mean error rate (%)'
    _log(f'\n{"="*60}', report)
    _log('ANALYSIS 3: Zhang et al. (2022) patterns — Rep-4', report)
    _log(f'Metric: {metric_label}', report)
    _log(f'{"="*60}\n', report)

    plot_rows = []

    for seq_name in _ZHANG_SEQ_NAMES:
        sub = df[(df['seq_name'] == seq_name) & (df['geom_tag'].isin(_ZHANG_TAGS))]
        if sub.empty:
            _log(f'[{seq_name}]: no data with zhang patterns.', report)
            continue

        tag_agg = {}
        for tag in _ZHANG_TAGS:
            agg = _aggregate_per_participant(sub[sub['geom_tag'] == tag], dl=dl)
            if len(agg) >= 3:
                tag_agg[tag] = agg

        _log(f'[{seq_name}]', report)
        for tag, agg in tag_agg.items():
            vals = list(agg.values())
            _log(f'  {tag}: n={len(vals)}, mean={round(np.mean(vals),3)}, '
                 f'SEM={round(float(stats.sem(vals)),3)}', report)
            plot_rows.append((seq_name, tag, float(np.mean(vals)), float(stats.sem(vals))))

        if len(tag_agg) >= 2:
            groups = [np.array(list(v.values())) for v in tag_agg.values()]
            stat_k, p_k = kruskal(*groups)
            _log(f'  Kruskal-Wallis H = {round(stat_k,3)}, p = {round(p_k,4)}, '
                 f'n groups = {[len(g) for g in groups]}', report)

            tag_names = list(tag_agg.keys())
            for i in range(len(tag_names)):
                for j in range(i + 1, len(tag_names)):
                    t1, t2 = tag_names[i], tag_names[j]
                    result = _pairwise_mwu(tag_agg, t1, t2)
                    if result:
                        stat_u, p_u, r_u, n1, n2 = result
                        _log(f'  Mann-Whitney {t1} vs {t2}: U={round(stat_u,1)}, '
                             f'p={round(p_u,4)}, r={round(r_u,3)}, n=({n1},{n2})', report)
        _log('', report)

    if not plot_rows:
        return

    seq_positions = {name: i for i, name in enumerate(_ZHANG_SEQ_NAMES)}
    plt.rcParams['figure.facecolor'] = 'white'
    fig, ax = plt.subplots(figsize=(8, 4))

    for seq_name, tag, mean, sem in plot_rows:
        y = seq_positions.get(seq_name, 0) + _ZHANG_JITTER.get(tag, 0)
        ax.errorbar(mean, y, xerr=sem, fmt='o', capsize=5,
                    linewidth=bar_frame_width,
                    color=_ZHANG_COLORS.get(tag, 'grey'),
                    label=tag if seq_name == _ZHANG_SEQ_NAMES[0] else '')

    ax.set_yticks(list(seq_positions.values()))
    ax.set_yticklabels(list(seq_positions.keys()), fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(metric_label, fontsize=title_size, labelpad=padding_size)
    ax.set_title('Zhang et al. (2022) patterns — Rep-4', fontsize=title_size)
    handles = [plt.Line2D([0], [0], marker='o', color=_ZHANG_COLORS[t],
                           linestyle='', label=t) for t in _ZHANG_TAGS]
    ax.legend(handles=handles, loc='lower right', fontsize=10, title='Pattern')
    plt.tight_layout()

    if path:
        fname = os.path.join(path, 'geometry_zhang_patterns.pdf')
        plt.savefig(fname, bbox_inches='tight', dpi=_adaptive_dpi())
        _log(f'Figure saved: {fname}', report)
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Visualisation: per-sequence delta DL vs starting vertex
# ---------------------------------------------------------------------------

def plot_starting_vertex_delta(df, seq_names=None, path=None, dl=True, report=None):
    """
    For each sequence, plot mean performance per starting vertex as a delta
    relative to vertex 0, with a shared Y-axis across all sequences.

    One PNG per sequence saved to path/starting_point_{seq_name}.png.

    Parameters
    ----------
    df        : DataFrame already processed with add_geom_columns().
    seq_names : list of str, optional.
    path      : str — directory to save PNGs. None = plt.show().
    dl        : bool — if True use DL distance, else error rate (%).
    report    : open file object, optional.
    """
    if seq_names is None:
        seq_names = seq_name_list

    metric_col   = 'distance_dl' if dl else '_error_pct'
    y_label      = 'Δ Distance DL (vs vertex 0)' if dl else 'Δ Error rate % (vs vertex 0)'
    title_prefix = 'Δ Distance DL'               if dl else 'Δ Error rate (%)'

    # ── Pass 1: compute stats and determine shared Y bounds ──────────────────
    plot_data_all = {}
    global_min = float('inf')
    global_max = float('-inf')

    for name in seq_names:
        sub = df[df['seq_name'] == name].copy()
        if sub.empty:
            continue

        if not dl:
            sub['_error_pct'] = (sub['performance'] != 'success').astype(float) * 100

        grp = (
            sub.groupby('starting_vertex')[metric_col]
            .agg(['mean', 'sem'])
            .reset_index()
        )

        baseline = grp.loc[grp['starting_vertex'] == 0, 'mean'].values
        if len(baseline) == 0:
            _log(f'  Warning: vertex 0 not found for {name}, skipping.', report)
            grp['delta_mean'] = grp['mean']
        else:
            grp['delta_mean'] = grp['mean'] - baseline[0]
            grp = grp[grp['starting_vertex'] != 0].copy()

        if grp.empty:
            continue

        plot_data_all[name] = grp
        global_min = min(global_min, (grp['delta_mean'] - grp['sem']).min())
        global_max = max(global_max, (grp['delta_mean'] + grp['sem']).max())

    if not plot_data_all:
        return

    y_range  = global_max - global_min
    y_limits = (global_min - 0.1 * y_range, global_max + 0.1 * y_range)

    # ── Pass 2: one figure per sequence ─────────────────────────────────────
    plt.rcParams['figure.facecolor'] = 'white'
    for name, grp in plot_data_all.items():
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.axhline(0, color='gray', linestyle='-', alpha=0.7, linewidth=1.2)
        ax.errorbar(
            x=grp['starting_vertex'],
            y=grp['delta_mean'],
            yerr=grp['sem'],
            fmt='-o',
            capsize=5,
            capthick=1.5,
            color='steelblue',
        )

        ax.set_title(f'{title_prefix} (relative to vertex 0):\n{name}',
                     fontsize=title_size)
        ax.set_xlabel('Starting vertex', fontsize=title_size, labelpad=padding_size)
        ax.set_ylabel(y_label, fontsize=title_size, labelpad=padding_size)
        ax.set_ylim(y_limits)
        ax.set_xticks(range(1, 6))
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if path:
            fname = os.path.join(path, f'starting_point_{name}.png')
            plt.savefig(fname, bbox_inches='tight', dpi=280)
        else:
            plt.show()
        plt.close()

    if path:
        _log(f'Starting vertex delta plots saved to: {path}', report)


# ---------------------------------------------------------------------------
# Top-level wrapper: run all geometry analyses for one experiment
# ---------------------------------------------------------------------------

def run_geometry_analysis(df, fig_dir, seq_names=None, dl=True, exp_label=''):
    """
    Run all three geometry analyses (starting vertex, shape, Zhang patterns),
    save figures as PDFs, and write a stats report to fig_dir.

    Parameters
    ----------
    df : DataFrame already processed with add_geom_columns().
    fig_dir : str — output directory for figures and report.
    seq_names : list of str, optional — sequences to include in analyses 1 & 2.
                Defaults to all available sequence names in df.
    dl : bool — if True report DL distance, else error rate.
    exp_label : str — label used in report header (e.g. 'Experiment 1').
    """
    os.makedirs(fig_dir, exist_ok=True)

    date_str = datetime.today().strftime('%d-%m-%Y')
    report_path = os.path.join(fig_dir, f'{date_str}_geometry_report.txt')

    with open(report_path, 'w') as report:
        header = (
            f'Geometry Effect Analysis — {exp_label}\n'
            f'Generated: {datetime.today().strftime("%Y-%m-%d %H:%M")}\n'
            f'Output directory: {fig_dir}\n'
            f'{"="*60}\n'
        )
        _log(header, report)

        if seq_names is None:
            seq_names = sorted(df['seq_name'].unique().tolist())

        # Print geom_tag distribution for reference
        _log('Geom tag distribution in data:', report)
        dist = df[df['seq_name'].isin(seq_names)]['geom_tag'].value_counts()
        for tag, count in dist.items():
            _log(f'  {tag}: {count}', report)
        _log('', report)

        test_starting_vertex_effect(df, seq_names=seq_names,
                                    path=fig_dir, dl=dl, report=report)
        test_shape_effect(df, seq_names=seq_names,
                          path=fig_dir, dl=dl, report=report)
        test_zhang_patterns(df, path=fig_dir, dl=dl, report=report)
        plot_starting_vertex_delta(df, seq_names=seq_names,
                                   path=fig_dir, dl=dl, report=report)

    print(f'\nReport saved: {report_path}')
