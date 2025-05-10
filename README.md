# Epistemic Network Analysis (ENA)

This project generates synthetic data for Epistemic Network Analysis and performs basic ENA on the generated data.

## Files
- `generate_ena_data.py`: Script to generate synthetic ENA data
- `perform_ena_analysis.py`: Script to perform basic ENA on the generated data
- `data_example.csv`: Example of the data format
- `ena_dataset.csv`: Generated dataset (created after running `generate_ena_data.py`)

## Data Structure
Each row in the dataset represents a stanza in Epistemic Network Analysis with the following columns:
- `Region`: Either "guangdong" or "hongkong"
- `documentid`: Document identifier in the format "meeting{i}"
- `stanzaid`: Unique identifier for each stanza within a document
- `stage`: One of "stage1", "stage2", or "stage3"
- Code columns (`ZA`, `HJ`, `ZC`, `GM`, `HX`, `LX`, `JN`, `XS`, `ZZ`): Binary indicators (0 or 1) showing whether the code occurs in the stanza

## Setup and Usage

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Generate the synthetic ENA data:
```
python generate_ena_data.py
```

3. Perform the ENA analysis:
```
python perform_ena_analysis.py
```

## Output
The analysis will produce:
- `adjacency_matrices.png`: Heatmap visualizations of the mean adjacency matrices for each region and stage
- `pca_visualization.png`: PCA visualization of the epistemic networks
- `network_graphs.png`: Network graph visualizations for each region and stage

## What is Epistemic Network Analysis?
Epistemic Network Analysis (ENA) is a method for identifying and quantifying connections among elements in coded data. It was developed to model the structure of connections in discourse, particularly for analyzing complex thinking and collaborative problem solving.

ENA creates network models by:
1. Identifying co-occurrences of codes within specified units of analysis (in this case, stanzas within documents)
2. Creating adjacency matrices to represent the connections between codes
3. Using dimensionality reduction to represent networks in a low-dimensional space
4. Visualizing the resulting networks to show how codes are connected 
 
