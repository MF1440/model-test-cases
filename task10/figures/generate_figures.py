#!/usr/bin/env python3
"""Generate figures for the draft.

Usage: python3 generate_figures.py
Reads: ../tests/search_log.csv, ../tests/coloring.txt
Output: fig1-fig6 PDF files in the same directory.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_LOG = os.path.join(OUTDIR, '..', 'tests', 'search_log.csv')
COLORING_FILE = os.path.join(OUTDIR, '..', 'tests', 'coloring.txt')

GOLDEN = (1 + np.sqrt(5)) / 2
R_KM = 6378.14
N = 64000
K = 1008


def fibonacci_spiral(n):
    """Fibonacci spiral on unit sphere, matches the solver."""
    i = np.arange(n, dtype=np.float64)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi = 2 * np.pi * i / GOLDEN
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def read_search_log():
    if not os.path.isfile(SEARCH_LOG):
        print(f'  [skip] {SEARCH_LOG} not found')
        return None
    rows = []
    with open(SEARCH_LOG, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'phase':       int(row['phase']),
                'round':       int(row['round']),
                'point':       int(row['point']),
                'distance_km': float(row['distance_km']),
                'edges':       int(row['edges']),
                'conflicts':   int(row['conflicts']),
                'feasible':    int(row['feasible']),
                'time_s':      float(row['time_s']),
            })
    return rows if rows else None


def read_coloring():
    if not os.path.isfile(COLORING_FILE):
        print(f'  [skip] {COLORING_FILE} not found')
        return None
    colors = []
    with open(COLORING_FILE) as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(',') if v.strip()]
            colors.extend(vals)
    return np.array(colors, dtype=np.int32)


def savefig(name):
    path = os.path.join(OUTDIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


# =============================================================================
# Fig 1: Fibonacci spiral on sphere
# =============================================================================
def fig1_fibonacci_spiral():
    print('Fig 1: Fibonacci spiral ...')
    x, y, z = fibonacci_spiral(N)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.viridis((z + 1) / 2)
    step = 8
    ax.scatter(x[::step], y[::step], z[::step], c=colors[::step], s=0.5, alpha=0.6)
    ax.set_title('Fibonacci Spiral (8,000 of 64,000 points shown)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    savefig('fig1_fibonacci_spiral.pdf')


# =============================================================================
# Fig 2: Search convergence from search_log.csv
# =============================================================================
def fig2_convergence():
    print('Fig 3: Convergence ...')
    data = read_search_log()
    if data is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for phase, color, label in [(1, '#2166ac', 'Phase 1 (CO)'),
                                 (2, '#b2182b', 'Phase 2 (TabuCol)')]:
        feasible = sorted(
            [r for r in data if r['phase'] == phase and r['feasible'] == 1],
            key=lambda r: r['time_s'])
        if not feasible:
            continue
        times, best = [], []
        cur = 0.0
        for r in feasible:
            if r['distance_km'] > cur:
                cur = r['distance_km']
            times.append(r['time_s'] / 60.0)
            best.append(cur)
        ax.step(times, best, where='post', color=color, linewidth=2, label=label)

    all_f = [r for r in data if r['feasible'] == 1]
    if all_f:
        best_r = max(all_f, key=lambda r: r['distance_km'])
        ax.plot(best_r['time_s'] / 60.0, best_r['distance_km'], 'k*',
                markersize=14, zorder=5,
                label=f'Best: {best_r["distance_km"]:.1f} km')

    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Best Feasible Distance (km)', fontsize=12)
    ax.set_title('Search Convergence: Multi-Point Stochastic Search', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    savefig('fig2_convergence.pdf')


# =============================================================================
# Fig 3: Conflicts vs distance from search_log.csv
# =============================================================================
def fig3_conflicts():
    print('Fig 3: Conflicts vs distance ...')
    data = read_search_log()
    if data is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Markers by method, colors by feasibility
    configs = [
        (1, True,  'o', '#2ecc71', 'CO feasible'),
        (1, False, 'o', '#e74c3c', 'CO infeasible'),
        (2, True,  '^', '#2ecc71', 'TabuCol feasible'),
        (2, False, '^', '#e74c3c', 'TabuCol infeasible'),
    ]
    for phase, is_feas, marker, color, label in configs:
        pts = [r for r in data
               if r['phase'] == phase and (r['conflicts'] == 0) == is_feas]
        if not pts:
            continue
        d = [r['distance_km'] for r in pts]
        c = [max(r['conflicts'], 1) for r in pts]
        ax.scatter(d, c, marker=marker, c=color, s=30, alpha=0.7,
                   edgecolors='k' if is_feas else 'none',
                   linewidths=0.4, label=label)

    ax.set_yscale('log')
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Conflicts', fontsize=12)
    ax.set_title('Conflicts vs Distance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    savefig('fig3_conflicts.pdf')


# =============================================================================
# Fig 4: Graph properties
# =============================================================================
def fig4_graph_properties():
    print('Fig 5: Graph properties ...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    d_range = np.linspace(100, 3500, 200)
    theta = 2 * np.arcsin(np.clip(d_range / (2 * R_KM), 0, 1))
    frac = (1 - np.cos(theta)) / 2
    avg_degree = N * frac

    # Left: degree vs distance
    ax1.plot(d_range, avg_degree, color='#3498db', linewidth=2.5)
    ax1.axhline(y=K, color='#e74c3c', linestyle='--', linewidth=1.5,
                label=f'K = {K:,} colors')
    ax1.axvline(x=2714, color='orange', linestyle=':', linewidth=1.5,
                label='Result (2,714 km)')
    ax1.fill_between(d_range, avg_degree, K, where=(avg_degree < K),
                      alpha=0.15, color='#2ecc71', label='deg < K (easy)')
    ax1.fill_between(d_range, avg_degree, K, where=(avg_degree >= K),
                      alpha=0.15, color='#e74c3c', label='deg >= K (hard)')
    ax1.set_xlabel('Chord Distance (km)', fontsize=12)
    ax1.set_ylabel('Average Vertex Degree', fontsize=12)
    ax1.set_title('(a) Graph Density vs Distance', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlim(0, 3500)
    ax1.set_ylim(0, 8000)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: edges vs distance
    edges = N * avg_degree / 2
    ax2.plot(d_range, edges / 1e6, color='#9b59b6', linewidth=2.5)
    # Result crosshair: vertical + horizontal lines
    idx_res = np.argmin(np.abs(d_range - 2714))
    e_res = edges[idx_res] / 1e6
    ax2.axvline(x=2714, color='orange', linestyle=':', linewidth=1.5,
                label='Result (2,714 km)')
    ax2.axhline(y=e_res, color='orange', linestyle=':', linewidth=1.0,
                alpha=0.7)
    ax2.plot(2714, e_res, 'o', color='orange', markersize=6, zorder=5)
    ax2.annotate(f'2,714 km — {e_res:.1f}M edges',
                xy=(2714, e_res), xytext=(1400, e_res + 20),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, ha='center', fontweight='bold')
    ax2.set_xlabel('Chord Distance (km)', fontsize=12)
    ax2.set_ylabel('Edges (millions)', fontsize=12)
    ax2.set_title('(b) Conflict Graph Size vs Distance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 3500)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    savefig('fig4_graph_properties.pdf')


# =============================================================================
# Fig 5: Color class + bottleneck pair on sphere
# =============================================================================
def fig5_coloring():
    print('Fig 5: Coloring ...')
    colors = read_coloring()
    if colors is None:
        return

    n = len(colors)
    x, y, z = fibonacci_spiral(n)
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    lon = np.arctan2(y, x)

    # Find the bottleneck pair: min chord distance between same-colored points
    R_m = R_KM * 1000.0
    best_dist = 1e18
    best_i, best_j = 0, 0
    best_color = 1
    # Check each color class
    for c in range(1, K + 1):
        idx = np.where(colors == c)[0]
        if len(idx) < 2:
            continue
        # Pairwise chord distances within this class (on unit sphere)
        px, py, pz = x[idx], y[idx], z[idx]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                d2 = (px[a]-px[b])**2 + (py[a]-py[b])**2 + (pz[a]-pz[b])**2
                if d2 < best_dist:
                    best_dist = d2
                    best_i, best_j = idx[a], idx[b]
                    best_color = c

    min_chord_km = np.sqrt(best_dist) * R_m / 1000.0

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='mollweide')

    # 8 highlighted color classes, roughly uniform across 1..1008
    show_colors = sorted([85, 186, 298, 402, 497, best_color, 724, 891])
    palette = ['#e74c3c', '#2ecc71', '#3498db', '#e67e22',
               '#9b59b6', '#1abc9c', '#f39c12', '#8e44ad']
    show_set = set(show_colors)

    # ~120 other color classes shown as gray (real cells)
    n_gray_classes = 120
    rng_fig = np.random.RandomState(42)
    all_colors_available = [c for c in range(1, K + 1) if c not in show_set]
    gray_classes = rng_fig.choice(all_colors_available,
                                   size=min(n_gray_classes, len(all_colors_available)),
                                   replace=False)
    gray_idx = np.concatenate([np.where(colors == c)[0] for c in gray_classes])
    ax.scatter(lon[gray_idx], lat[gray_idx], s=4, c='0.85', alpha=0.6,
               edgecolors='0.5', linewidths=0.1)

    # Highlighted color classes
    for sc, pc in zip(show_colors, palette):
        idx_c = np.where(colors == sc)[0]
        if len(idx_c) < 2:
            continue
        ax.scatter(lon[idx_c], lat[idx_c], s=10, c=pc, alpha=0.9,
                   edgecolors='k', linewidths=0.2, zorder=3,
                   label=f'Color {sc} ({len(idx_c)} pts)')

    # Minimum-distance edge
    ax.plot([lon[best_i], lon[best_j]], [lat[best_i], lat[best_j]],
            color='darkred', linewidth=0.8, zorder=4)

    # Annotate with distance
    mid_lon = (lon[best_i] + lon[best_j]) / 2
    mid_lat = (lat[best_i] + lat[best_j]) / 2
    ax.annotate(f'{min_chord_km:.1f} km',
                xy=(mid_lon, mid_lat), fontsize=10, fontweight='bold',
                color='darkred', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='darkred', alpha=0.9),
                zorder=6)

    n_shown = len(gray_idx) + sum(
        len(np.where(colors == sc)[0]) for sc in show_colors)
    ax.set_title(f'{K} colors '
                 f'— minimum same-color separation {min_chord_km:.1f} km\n'
                 f'({n_shown:,} of {n:,} cells shown: 8 colored + {n_gray_classes} gray)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='upper right', frameon=True, ncol=2,
              markerscale=1.5)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    savefig('fig5_coloring.pdf')


# =============================================================================
if __name__ == '__main__':
    print('=' * 50)
    print('Generating figures')
    print('=' * 50)
    fig1_fibonacci_spiral()
    fig2_convergence()
    fig3_conflicts()
    fig4_graph_properties()
    fig5_coloring()
    print('=' * 50)
    print('Done.')
