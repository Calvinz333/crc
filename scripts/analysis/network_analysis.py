 """
Microbial Co-occurrence Network Analysis
========================================
Builds co-occurrence networks for CRC and CTR groups separately, 
highlighting differential abundance status.

Steps:
  - Select features: Top 80 DA taxa (based on Cohen's d) + top 70 most variant taxa.
  - Compute Spearman correlation across the two groups (CRC vs CTR).
  - Apply significance threshold (FDR < 0.05) and |Spearman rho| > 0.3.
  - Build NetworkX graphs.
  - Export network topologies.
  - Plot and save networks.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import warnings, os
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ML_DIR  = "data/processed/ml_ready"
DA_DIR  = "results/differential"
RES_DIR = "results/network"
FIG_DIR = "figures/ml"
for d in [RES_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150})

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Microbial Co-occurrence Network Analysis")
print("=" * 65)
print("\n[1] Loading data...")

X = pd.read_csv(f"{ML_DIR}/X_species_combat.csv", index_col=0)
y = pd.read_csv(f"{ML_DIR}/y_labels.csv", index_col=0).squeeze()

try:
    da_res = pd.read_csv(f"{DA_DIR}/da_results.csv")
    da_sig = da_res[da_res['qval'] < 0.05]
    print(f"    Loaded DA results: {len(da_sig)} significant features.")
except:
    print("    Could not load DA results. Run differential_abundance.py first.")
    exit(1)

# Feature selection for network mapping to keep node count interpretable (~150 nodes)
sig_taxa  = da_res[da_res['qval']<0.05].nlargest(60, 'cohens_d')['species'].tolist()
sig_taxa += da_res[da_res['qval']<0.05].nsmallest(60, 'cohens_d')['species'].tolist()

top_var   = X.var().nlargest(100).index.tolist()
selected_features = list(set(sig_taxa + top_var))[:150]

X_sel = X[selected_features]
crc_idx = y[y == 1].index
ctr_idx = y[y == 0].index

print(f"    Selected {len(selected_features)} features for network construction.")

# DA Status Dict for node colouring
da_status = {}
for feat in selected_features:
    row = da_res[da_res['species'] == feat]
    if len(row) > 0 and row.iloc[0]['qval'] < 0.05:
        if row.iloc[0]['log2fc'] > 0:
            da_status[feat] = 'CRC-enriched'
        else:
            da_status[feat] = 'CTR-enriched'
    else:
        da_status[feat] = 'NS'

# ══════════════════════════════════════════════════════════════════════════
# 2. COMPUTE CORRELATION & BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════
def build_network(df, corr_thresh=0.35, pval_thresh=0.01):
    """Compute Spearman corr, apply FDR, and return NetworkX graph."""
    feats = df.columns
    n = len(feats)
    
    # Calculate Spearman correlation matrix and p-values
    rho, pval = spearmanr(df.values, axis=0)
    
    # Flatten upper triangle
    edges = []
    pvals = []
    
    for i in range(n):
        for j in range(i+1, n):
            edges.append((feats[i], feats[j], rho[i, j]))
            pvals.append(pval[i, j])
            
    # FDR Correction on edge p-values
    reject, qvals, _, _ = multipletests(pvals, method='fdr_bh')
    
    G = nx.Graph()
    for node in feats:
        G.add_node(node, da_status=da_status[node])
        
    for k, (u, v, r) in enumerate(edges):
        if reject[k] and qvals[k] < pval_thresh and abs(r) >= corr_thresh:
            v_type = 'positive' if r > 0 else 'negative'
            G.add_edge(u, v, weight=r, type=v_type)
            
    # Remove isolated nodes to make plots cleaner
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

print("\n[2] Building networks (Spearman rho ≥ 0.35, FDR < 0.01)...")
G_crc = build_network(X_sel.loc[crc_idx], corr_thresh=0.35, pval_thresh=0.01)
print(f"    CRC Network : {G_crc.number_of_nodes()} nodes, {G_crc.number_of_edges()} edges")

G_ctr = build_network(X_sel.loc[ctr_idx], corr_thresh=0.35, pval_thresh=0.01)
print(f"    CTR Network : {G_ctr.number_of_nodes()} nodes, {G_ctr.number_of_edges()} edges")

# Export Edge Lists
nx.write_edgelist(G_crc, f"{RES_DIR}/crc_network_edges.csv", delimiter=",", data=['weight', 'type'])
nx.write_edgelist(G_ctr, f"{RES_DIR}/ctr_network_edges.csv", delimiter=",", data=['weight', 'type'])

# ══════════════════════════════════════════════════════════════════════════
# 3. GLOBAL TOPOLOGY PROPERTIES
# ══════════════════════════════════════════════════════════════════════════
def get_props(G, name):
    degrees = [d for n, d in G.degree()]
    return {
        'Network': name,
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Density': round(nx.density(G), 4),
        'Avg Degree': round(np.mean(degrees) if degrees else 0, 2),
        'Avg Clustering': round(nx.average_clustering(G) if G.number_of_nodes()>0 else 0, 4)
    }

props = [get_props(G_crc, 'CRC'), get_props(G_ctr, 'CTR')]
df_props = pd.DataFrame(props)
df_props.to_csv(f"{RES_DIR}/network_properties.csv", index=False)
print("\n[3] Network Topologies:")
print(df_props.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# 4. PLOTTING NETWORKS
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] Generating network plots...")

def plot_net(G, title, filename):
    if G.number_of_nodes() == 0:
        print(f"     Skipping {title} plot (0 nodes).")
        return
        
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    # Node Colors
    color_map = {'CRC-enriched': '#E63946', 'CTR-enriched': '#457B9D', 'NS': '#B0BEC5'}
    node_colors = [color_map[G.nodes[n].get('da_status', 'NS')] for n in G.nodes()]
    
    # Edge Colors & Width
    edge_colors = ['#FF8A80' if G.edges[e]['type'] == 'positive' else '#82B1FF' for e in G.edges()]
    edge_weights = [abs(G.edges[e]['weight']) * 2.5 for e in G.edges()]
    
    # Draw Graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=60, alpha=0.8, edgecolors='white', linewidths=0.5)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, alpha=0.6)
    
    # Labels for top 5 degree nodes (hub nodes)
    degrees = dict(G.degree())
    if len(degrees) > 0:
        top_hubs = sorted(degrees, key=degrees.get, reverse=True)[:5]
        labels = {n: n.split('[')[0].strip()[:20] for n in top_hubs}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', font_color='#2c3e50')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E63946', markersize=8, label='CRC-enriched Taxa'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#457B9D', markersize=8, label='CTR-enriched Taxa'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#B0BEC5', markersize=8, label='Non-significant Taxa'),
        Line2D([0], [0], color='#FF8A80', lw=2, label='Positive Correlation (+)'),
        Line2D([0], [0], color='#82B1FF', lw=2, label='Negative Correlation (–)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename}", bbox_inches='tight', dpi=200)
    plt.close()
    print(f"    Saved: {FIG_DIR}/{filename}")

plot_net(G_crc, "Microbial Co-occurrence Network — CRC Group", "08_network_crc.png")
plot_net(G_ctr, "Microbial Co-occurrence Network — CTR Group", "09_network_ctr.png")

print("\n" + "=" * 65)
print("  NETWORK ANALYSIS COMPLETE ✅")
print("=" * 65)
