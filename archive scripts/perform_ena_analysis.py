import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sklearn.manifold import MDS

# Load the data
def load_data(file_path):
    print("Loading data from:", file_path)
    return pd.read_csv(file_path)

# Create adjacency matrices for each unit (document+stanza)
def create_adjacency_matrices(df, codes):
    print("Creating adjacency matrices...")
    
    # Get unique units (region, document, stage combinations)
    units = df[['Region', 'documentid', 'stage']].drop_duplicates().reset_index(drop=True)
    
    # Create a dictionary to store adjacency matrices
    adj_matrices = {}
    
    # For each unit, create an adjacency matrix
    for i, unit in units.iterrows():
        region = unit['Region']
        document = unit['documentid']
        stage = unit['stage']
        
        # Filter dataframe for this unit
        unit_df = df[(df['Region'] == region) & 
                     (df['documentid'] == document) & 
                     (df['stage'] == stage)]
        
        # Create adjacency matrix for codes
        adj_matrix = np.zeros((len(codes), len(codes)))
        
        # For each stanza in this unit
        for _, stanza in unit_df.iterrows():
            # Get codes present in this stanza
            present_codes = [code for code in codes if stanza[code] == 1]
            
            # For each pair of present codes, increment adjacency matrix
            for i, code1 in enumerate(present_codes):
                for j, code2 in enumerate(present_codes):
                    if i != j:  # Don't count self-connections
                        code1_idx = codes.index(code1)
                        code2_idx = codes.index(code2)
                        adj_matrix[code1_idx, code2_idx] += 1
        
        # Store adjacency matrix
        adj_matrices[(region, document, stage)] = adj_matrix
    
    return adj_matrices, units

# Calculate mean adjacency matrices
def calculate_mean_adjacency(adj_matrices, units, codes):
    print("Calculating mean adjacency matrices...")
    
    # Calculate mean adjacency matrix for each region and stage
    region_stage_means = {}
    
    for region in units['Region'].unique():
        for stage in units['stage'].unique():
            # Get all units for this region and stage
            region_stage_units = [(r, d, s) for (r, d, s) in adj_matrices.keys() 
                                if r == region and s == stage]
            
            # Sum adjacency matrices
            sum_adj = np.zeros((len(codes), len(codes)))
            for unit in region_stage_units:
                sum_adj += adj_matrices[unit]
            
            # Calculate mean
            mean_adj = sum_adj / len(region_stage_units)
            region_stage_means[(region, stage)] = mean_adj
    
    return region_stage_means

# Visualize mean adjacency matrices
def visualize_adjacency_matrices(mean_adj_matrices, codes):
    print("Visualizing adjacency matrices...")
    
    # Set up matplotlib figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # For each region-stage combination
    for i, ((region, stage), adj_matrix) in enumerate(mean_adj_matrices.items()):
        # Create heatmap
        sns.heatmap(adj_matrix, annot=True, cmap="YlGnBu", 
                   xticklabels=codes, yticklabels=codes, ax=axes[i])
        axes[i].set_title(f"{region} - {stage}")
    
    plt.tight_layout()
    plt.savefig("adjacency_matrices.png")
    print("Saved adjacency matrix visualization to adjacency_matrices.png")

# Perform PCA on adjacency matrices
def perform_pca(adj_matrices, units):
    print("Performing PCA on adjacency matrices...")
    
    # Flatten adjacency matrices into vectors
    X = []
    labels = []
    
    for (region, document, stage), adj_matrix in adj_matrices.items():
        # Flatten upper triangle of adjacency matrix (excluding diagonal)
        flat_matrix = []
        for i in range(adj_matrix.shape[0]):
            for j in range(i+1, adj_matrix.shape[1]):
                flat_matrix.append(adj_matrix[i, j])
        
        X.append(flat_matrix)
        labels.append((region, document, stage))
    
    # Convert to numpy array
    X = np.array(X)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create dataframe with PCA results
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Region': [label[0] for label in labels],
        'document': [label[1] for label in labels],
        'stage': [label[2] for label in labels]
    })
    
    return pca_df, pca.explained_variance_ratio_

# Visualize PCA results
def visualize_pca(pca_df, explained_variance):
    print("Visualizing PCA results...")
    
    # Plot PCA results
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    sns.scatterplot(
        data=pca_df, x='PC1', y='PC2', hue='Region', style='stage', s=100
    )
    
    plt.title('PCA of Epistemic Networks')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance explained)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance explained)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("pca_visualization.png")
    print("Saved PCA visualization to pca_visualization.png")

# Visualize code embeddings in 2D space for each stage
def visualize_code_embeddings(mean_adj_matrices, codes):
    print("Visualizing code embeddings for each stage...")
    
    # Set up figure with 3 subplots (one for each stage)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    stages = sorted(set(stage for (_, stage) in mean_adj_matrices.keys()))
    
    # Color map for regions
    region_colors = {"guangdong": "blue", "hongkong": "red"}
    
    # For each stage
    for i, stage in enumerate(stages):
        # Get all regions for this stage
        regions = sorted(set(region for (region, s) in mean_adj_matrices.keys() if s == stage))
        
        # For each region in this stage
        for region in regions:
            # Get adjacency matrix for this region and stage
            adj_matrix = mean_adj_matrices[(region, stage)]
            
            # Use MDS to project the codes into 2D space
            # Convert adjacency matrix to distance matrix (higher value = closer)
            # We use 1 - normalized adjacency for distance
            adj_max = adj_matrix.max()
            if adj_max > 0:  # Avoid division by zero
                distance_matrix = 1 - (adj_matrix / adj_max)
                np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0
            else:
                distance_matrix = 1 - adj_matrix  # All zeros anyway
            
            # Apply MDS
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            code_coordinates = mds.fit_transform(distance_matrix)
            
            # Plot code positions
            axes[i].scatter(
                code_coordinates[:, 0], 
                code_coordinates[:, 1],
                alpha=0.7,
                label=region,
                color=region_colors[region],
                s=100
            )
            
            # Label each point with its code name
            for j, code in enumerate(codes):
                axes[i].annotate(
                    code,
                    (code_coordinates[j, 0], code_coordinates[j, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color=region_colors[region]
                )
        
        axes[i].set_title(f"Code Embeddings - {stage}")
        axes[i].legend()
        axes[i].set_xlabel("Dimension 1")
        axes[i].set_ylabel("Dimension 2")
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("code_embeddings.png")
    print("Saved code embeddings visualization to code_embeddings.png")

# Visualize network graphs for each region and stage
def visualize_networks(mean_adj_matrices, codes):
    print("Visualizing network graphs...")
    
    # Set up matplotlib figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # For each region-stage combination
    for i, ((region, stage), adj_matrix) in enumerate(mean_adj_matrices.items()):
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for code in codes:
            G.add_node(code)
        
        # Add edges with weights
        for i_idx, code1 in enumerate(codes):
            for j_idx, code2 in enumerate(codes):
                if i_idx != j_idx and adj_matrix[i_idx, j_idx] > 0:
                    G.add_edge(code1, code2, weight=adj_matrix[i_idx, j_idx])
        
        # Set edge weights for visualization
        edge_weights = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]
        
        # Draw graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(
            G, pos, node_size=500, node_color='lightblue', ax=axes[i]
        )
        nx.draw_networkx_labels(G, pos, ax=axes[i])
        nx.draw_networkx_edges(
            G, pos, width=edge_weights, alpha=0.7, 
            edge_color='gray', arrows=True, ax=axes[i]
        )
        
        axes[i].set_title(f"{region} - {stage}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("network_graphs.png")
    print("Saved network graph visualization to network_graphs.png")

# Main function
def main():
    file_path = "ena_dataset.csv"
    df = load_data(file_path)
    
    # Get codes (columns that are not Region, documentid, stanzaid, or stage)
    codes = [col for col in df.columns 
            if col not in ['Region', 'documentid', 'stanzaid', 'stage']]
    
    # Create adjacency matrices
    adj_matrices, units = create_adjacency_matrices(df, codes)
    
    # Calculate mean adjacency matrices
    mean_adj_matrices = calculate_mean_adjacency(adj_matrices, units, codes)
    
    # Visualize mean adjacency matrices
    visualize_adjacency_matrices(mean_adj_matrices, codes)
    
    # Perform PCA
    pca_df, explained_variance = perform_pca(adj_matrices, units)
    
    # Visualize PCA results
    visualize_pca(pca_df, explained_variance)
    
    # Visualize networks
    visualize_networks(mean_adj_matrices, codes)
    
    # Visualize code embeddings
    visualize_code_embeddings(mean_adj_matrices, codes)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 