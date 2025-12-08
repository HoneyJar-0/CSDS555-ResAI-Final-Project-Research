import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame

from configs import evaluation_config

def load_df(filename):
    return pd.read_parquet(f"{evaluation_config.data_dir}/condensed/{filename}.parquet")

def get_control_vectors(df: pd.DataFrame):
    feature_columns = df.columns.difference(['identity_A', 'identity_B'])

    # Base control vector: identity_A=0 & identity_B=0
    mask_base = (df['identity_A'] == 0) & (df['identity_B'] == 0)
    control_base = df.loc[mask_base, feature_columns].to_numpy()
    if control_base.shape[0] > 0:
        control_base = control_base[0]
    else:
        control_base = np.array([])

    # Variable control: identity_A=0, variable B
    mask_A = df['identity_A'] == 0
    control_A = df.loc[mask_A, feature_columns].to_numpy()

    # Variable control: identity_B=0, variable A
    mask_B = df['identity_B'] == 0
    control_B = df.loc[mask_B, feature_columns].to_numpy()

    return control_base, control_A, control_B

def make_heatmap(df: DataFrame, condition, control_method="B"):
    df = df.drop(columns=["UUID", "model", "is_blocked"])
    feature_columns = df.columns.difference(['identity_A', 'identity_B'])

    control_base, control_A, control_B = get_control_vectors(df)

    features = df[feature_columns].to_numpy()
    identity_A = df['identity_A'].to_numpy(dtype=int)
    identity_B = df['identity_B'].to_numpy(dtype=int)

    if control_method == "center":
        features -= control_base
    elif control_method == "A":
        features -= control_A[identity_B.astype(int)]
    elif control_method == "B":
        features -= control_B[identity_A.astype(int)]

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


def plot_graph(graph, labels=None, edge_color='red'):
    plt.figure(figsize=(8, 6))

    nx.draw(graph,
            with_labels=True,
            labels=labels,          # If we want to add text to each identity
            node_color='lightblue', # Points color here
            node_size=500,
            font_size=10,
            font_weight='bold',
            edge_color=edge_color)       # Arrow colors here

    plt.title("Graph Visualization")
    plt.show()

if __name__ == "__main__":
    dataset = load_df(filename="llama_0_1406595")
    print(f"Loaded DataFrame: {len(dataset)}")
    print("Making heatmap visualization...")
    hmp, hm_df = make_heatmap(dataset, condition="positive", control_method="center")
    print(hm_df.head(10)) # heatmap: rows represent Identity A, columns represent Identity B <--- Important!!!!!!
    print("Generating Graph...")
    graph = generate_graph(hmp, threshold=0.40, bidirectional=True)
    print(f"Graph generated of {graph.number_of_nodes()} nodes.")
    plot_graph(graph, edge_color='green')