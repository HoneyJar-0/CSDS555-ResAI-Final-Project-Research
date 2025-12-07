import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame

from configs import evaluation_config

def load_df(filename):
    return pd.read_parquet(f"{evaluation_config.data_dir}/condensed/{filename}.parquet")

def get_control_vectors(df: DataFrame):
    # Base control vector --> When identity_A=0 and identity_B=0
    control_base = df[(df['identity_A'] == 0) & (df['identity_B'] == 0)]
    if not control_base.empty:
        control_base= control_base.drop(columns=['identity_A', 'identity_B']).values[0]
    else:
        control_base = torch.tensor([])

    # Variable control matrix --> When A=0 and variable B
    control_A = df[df['identity_A'] == 0]
    control_A = control_A.drop(columns=['identity_A']).drop(columns=['identity_B']).values

    # Variable control matrix --> When B=0 and variable A
    control_B = df[df['identity_B'] == 0]
    control_B = control_B.drop(columns=['identity_A']).drop(columns=['identity_B']).values

    return control_base.tolist(), control_A.tolist(), control_B.tolist()

def make_heatmap(df: DataFrame, condition, control_method="B"):
    # First, drop irrelevant columns
    df = df.drop(columns=["UUID", "model", "is_blocked"])
    feature_columns = df.columns.difference(['identity_A', 'identity_B'])

    # Next, use control set and subtract
    control_base, control_A, control_B = get_control_vectors(df)

    def subtract_row(row):
        feature_vector = row[feature_columns].values

        if control_method == "center":
            feature_vector -= control_base

        elif control_method == "A":
            control_value = control_A[int(row['identity_B'])]
            feature_vector -= control_value

        elif control_method == "B":
            control_value = control_B[int(row['identity_A'])]
            feature_vector -= control_value

        return pd.Series(feature_vector, index=feature_columns)
    df[feature_columns] = df.apply(subtract_row, axis=1)

    # Next, group by identity A, B
    df = df.groupby(['identity_A', 'identity_B'], as_index=False).mean()

    # Next, convert to heatmap & condition by a specific feature
    condition_idx = df.columns.get_loc(condition)

    identity_As = {identity: idx for idx, identity in enumerate(df['identity_A'].unique())}
    identity_Bs = {identity: idx for idx, identity in enumerate(df['identity_B'].unique())}

    heatmap = torch.zeros((len(identity_As), len(identity_Bs), len(df.columns)-2), dtype=torch.float)

    def insert(row):
        # Extract indices for identity_A and identity_B
        identity_A_idx = identity_As[row['identity_A']]
        identity_B_idx = identity_Bs[row['identity_B']]

        # Get all the values besides identities
        feature_vector = torch.tensor(row.drop(['identity_A', 'identity_B']).values, dtype=torch.float32)

        # Fill the heatmap
        heatmap[identity_A_idx, identity_B_idx] = feature_vector

    df.apply(insert, axis=1)

    # Convert into readable form (not needed)
    heatmap = heatmap[:, :, condition_idx-2].detach().numpy()
    df = pd.DataFrame(heatmap, index=list(identity_As.keys()), columns=list(identity_Bs.keys()))

    # Sanity check: Min and max make sense
    print(f"Heatmap Min: {heatmap.min()}, Max: {heatmap.max()}")

    return heatmap, df

def generate_graph(heatmap: np.ndarray, threshold: float = 0.5, bidirectional: bool = False):
    adjacency_matrix = (heatmap > threshold).astype(int)
    G = nx.DiGraph()

    if bidirectional:
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                # Check if both idA -> idB and idB -> idA are above the bias threshold
                if adjacency_matrix[i, j] == 1 and adjacency_matrix[j, i] == 1:
                    G.add_edge(i, j)
                    G.add_edge(j, i)
    else:
        # Works with lopsided matrix (should not happen when our dataset is fully done)
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] == 1:
                    G.add_edge(i, j)
    return G


def plot_graph(graph, labels=None):
    plt.figure(figsize=(8, 6))

    nx.draw(graph,
            with_labels=True,
            labels=labels,          # If we want to add text to each identity
            node_color='lightblue', # Points color here
            node_size=500,
            font_size=10,
            font_weight='bold',
            edge_color='red')       # Arrow colors here

    plt.title("Graph Visualization")
    plt.show()

if __name__ == "__main__":
    dataset = load_df(filename="llama_281320_381219")
    print("Making heatmap visualization...")
    hmp, hm_df = make_heatmap(dataset, condition="negative")
    print(hm_df.head(10)) # heatmap: rows represent Identity A, columns represent Identity B <--- Important!!!!!!
    print("Generating Graph...")
    graph = generate_graph(hmp, threshold=0.6)
    print(f"Graph generated of {graph.number_of_nodes()} nodes.")
    plot_graph(graph)