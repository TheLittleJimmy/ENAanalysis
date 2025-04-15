import pandas as pd
import random

# Set random seed for reproducibility
random.seed(42)

# Load the full dataset
full_data = pd.read_csv('ena_dataset.csv')

# Define the criteria for the subset
# Include both regions, all three stages, and 2 documents from each region-stage combination
selected_docs = []
for region in ['guangdong', 'hongkong']:
    for stage in ['stage1', 'stage2', 'stage3']:
        # Get all unique document IDs for this region and stage
        docs = full_data[(full_data['Region'] == region) & 
                         (full_data['stage'] == stage)]['documentid'].unique()
        
        # Randomly select 2 documents
        selected = random.sample(list(docs), min(2, len(docs)))
        selected_docs.extend(selected)

# Filter the data to only include the selected documents
subset = full_data[full_data['documentid'].isin(selected_docs)]

# For each document, only include a maximum of 20 stanzas
final_subset = []
for doc in selected_docs:
    doc_data = subset[subset['documentid'] == doc]
    
    # If the document has more than 20 stanzas, randomly sample 20
    if len(doc_data) > 20:
        sampled_stanzas = random.sample(list(doc_data['stanzaid'].unique()), 20)
        doc_subset = doc_data[doc_data['stanzaid'].isin(sampled_stanzas)]
    else:
        doc_subset = doc_data
    
    final_subset.append(doc_subset)

# Combine all subsets into a single DataFrame
final_subset_df = pd.concat(final_subset, ignore_index=True)

# Save the subset to a new CSV file
final_subset_df.to_csv('ena_dataset_subset.csv', index=False)

# Print some information about the subset
print(f"Full dataset size: {len(full_data)} rows")
print(f"Subset size: {len(final_subset_df)} rows")
print(f"Documents in subset: {len(final_subset_df['documentid'].unique())}")
print(f"Regions in subset: {final_subset_df['Region'].unique()}")
print(f"Stages in subset: {final_subset_df['stage'].unique()}")

# Show a sample of the subset
print("\nSample of the subset data:")
print(final_subset_df.head(10))