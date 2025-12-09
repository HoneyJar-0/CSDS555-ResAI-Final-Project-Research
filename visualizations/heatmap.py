import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from pandas import DataFrame

from configs import evaluation_config

def load_df(filename):
    return pd.read_parquet(f"{evaluation_config.data_dir}/condensed/{filename}.parquet")

def get_control_vectors(df: pd.DataFrame):
    feature_columns = df.columns.difference(['identity_A', 'identity_B'])
    features = df.loc[:, feature_columns].to_numpy()

    # Base control vector: identity_A=0 & identity_B=0
    mask_base = (df['identity_A'] == 0) & (df['identity_B'] == 0)
    control_base = df.loc[mask_base, feature_columns].to_numpy()
    if control_base.shape[0] > 0:
        control_base = control_base[0]
    else:
        control_base = np.array([])

    # Variable control: identity_A=0, variable B
    control_A = np.zeros_like(features)
    mask_A = df['identity_A'] == 0
    control_A[mask_A] = features[mask_A]

    # Variable control: identity_B=0, variable A
    control_B = np.zeros_like(features)
    mask_B = df['identity_B'] == 0
    control_B[mask_B] = features[mask_B]

    return control_base, control_A, control_B

def make_heatmap(df: DataFrame, condition, control_method="B"):
    df = df.drop(columns=["UUID", "model", "is_blocked"])
    feature_columns = df.columns.difference(['identity_A', 'identity_B'])

    control_base, control_A, control_B = get_control_vectors(df)

    features = df[feature_columns].to_numpy()

    if control_method == "center":
        features -= control_base
    elif control_method == "A":
        features -= control_A
    elif control_method == "B":
        features -= control_B

    df[feature_columns] = features
    df = df.groupby(['identity_A', 'identity_B'], as_index=False).mean()

    condition_idx = df.columns.get_loc(condition)
    identity_As = {identity: idx for idx, identity in enumerate(df['identity_A'].unique())}
    identity_Bs = {identity: idx for idx, identity in enumerate(df['identity_B'].unique())}

    identity_A_idx = df['identity_A'].map(identity_As).to_numpy()
    identity_B_idx = df['identity_B'].map(identity_Bs).to_numpy()
    feature_values = torch.tensor(df[feature_columns].to_numpy(), dtype=torch.float32)

    heatmap = torch.zeros((len(identity_As), len(identity_Bs), len(feature_columns)), dtype=torch.float32)
    heatmap[identity_A_idx, identity_B_idx] = feature_values

    heatmap_np = heatmap[:, :, condition_idx - 2].detach().numpy()
    df_heatmap = pd.DataFrame(heatmap_np, index=list(identity_As.keys()), columns=list(identity_Bs.keys()))

    print(f"Heatmap Min: {heatmap_np.min()}, Max: {heatmap_np.max()}")
    return heatmap_np, df_heatmap

def generate_graph(heatmap: np.ndarray, threshold: float = 0.5, bidirectional: bool = False):
    adjacency_matrix = (heatmap > threshold).astype(int)
    print(f"Shape: {adjacency_matrix.shape}")
    if not adjacency_matrix.shape[0] == adjacency_matrix.shape[1] and bidirectional:
        print("Could not use bidirectional graph since heatmap is non-square.")
        bidirectional = False

    if bidirectional:
        G = nx.Graph()
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                # Check if both idA -> idB and idB -> idA are above the bias threshold
                if adjacency_matrix[i, j] == 1 and adjacency_matrix[j, i] == 1:
                    G.add_edge(i, j)
                    G.add_edge(j, i)
    else:
        # Works with lopsided matrix (should not happen when our dataset is fully done)
        G = nx.DiGraph()
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] == 1:
                    G.add_edge(i, j)
    return G


def plot_graph(graph, labels=None, edge_color=None):
    # Seaborn used for good colors n stuff
    sns.set_style("darkgrid")
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(14, 9))
    pos = nx.spring_layout(graph, k=0.4, iterations=100)
    node_colors = sns.color_palette("husl", len(graph.nodes()))
    nx.draw_networkx_nodes(graph, pos,
                           node_size=600,
                           node_color=node_colors,
                           edgecolors='white',
                           linewidths=2,
                           alpha=0.9,
                           ax=ax)

    nx.draw_networkx_edges(graph, pos,
                           width=2,
                           alpha=0.6,
                           edge_color=edge_color,
                           style='solid',
                           ax=ax)

    nx.draw_networkx_labels(graph, pos,
                            font_size=11,
                            font_weight='bold',
                            font_color='white',
                            ax=ax)

    # Legend logic
    if labels is not None:
        legend_elements = []
        for i, (node_id, text) in enumerate(labels.items()):
            color = node_colors[i % len(node_colors)]  # Make it match color
            legend_elements.append(Patch(facecolor=color, edgecolor='white', linewidth=1.5, label=f"{node_id}: {text}"))

        legend = ax.legend(handles=legend_elements,
                           title="Node Labels",
                           title_fontsize=12,
                           fontsize=10,
                           loc='center left',
                           bbox_to_anchor=(1.05, 0.5),
                           frameon=True,
                           framealpha=0.95,
                           edgecolor='gray')
        legend.get_frame().set_facecolor('#f8f9fa')

    ax.set_facecolor('#f8f9fa')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset = load_df(filename="llama_0_1406595")
    print(f"Loaded DataFrame: {len(dataset)}")
    print("Making heatmap visualization...")
    hmp, hm_df = make_heatmap(dataset, condition="negative", control_method="A") # control_method says which variable to fix (make 'independent' in a way)

    # Filter heatmap to specific genders
    #hmp = hmp[:73, :73]

    print(hm_df.head(10)) # heatmap: rows represent Identity A, columns represent Identity B <--- Important!!!!!!
    print("Generating Graph...")
    graph = generate_graph(hmp, threshold=0.65, bidirectional=True)
    print(f"Graph generated of {graph.number_of_nodes()} nodes.")

    print(f"Setting up labels...")
    identities = pd.read_csv(f'{evaluation_config.input_dir}/identities.csv')
    ids = list(graph.nodes())
    labels = (identities[identities['id'].isin(ids)]['identity'].apply(lambda x: '-'.join([w[:4] for w in str(x).split()]))) # Substring to first 3 chars for readability
    plot_graph(graph, labels=labels, edge_color='red')