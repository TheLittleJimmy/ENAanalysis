import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('enadata_processed.csv')

# Check the columns
print("Columns:", df.columns.tolist())

# Print basic info about the data
print("\nData shape:", df.shape)

# Get unique codes
codes = df['code'].dropna().str.split(',').explode().unique()
print("\nUnique codes:", codes)

# Get unique stages
stages = df['stage'].unique()
print("\nUnique stages:", stages)

# Get unique regions
regions = df['Region'].unique()
print("\nUnique regions:", regions)

# Get unique document IDs
document_ids = df['document-id'].unique()
print("\nNumber of unique document IDs:", len(document_ids))

# 1. Create a table of stage vs. code
print("\n1. Creating stage vs. code table...")
stage_code_counts = pd.DataFrame(0, index=stages, columns=codes)

# Count occurrences of each code in each stage
for stage in stages:
    stage_data = df[df['stage'] == stage]
    # For rows with code, split and count each code
    for _, row in stage_data.iterrows():
        if isinstance(row['code'], str) and row['code'].strip():
            for code in row['code'].split(','):
                stage_code_counts.loc[stage, code.strip()] += 1

print(stage_code_counts)

# 2. Create a table of region vs. code
print("\n2. Creating region vs. code table...")
region_code_counts = pd.DataFrame(0, index=regions, columns=codes)

# Count occurrences of each code in each region
for region in regions:
    region_data = df[df['Region'] == region]
    # For rows with code, split and count each code
    for _, row in region_data.iterrows():
        if isinstance(row['code'], str) and row['code'].strip():
            for code in row['code'].split(','):
                region_code_counts.loc[region, code.strip()] += 1

print(region_code_counts)

# 3. Create a table of document-id vs. code
print("\n3. Creating document-id vs. code table...")
# Since there might be many document IDs, limit to top 15 by number of entries
top_docs = df['document-id'].value_counts().head(15).index.tolist()
doc_code_counts = pd.DataFrame(0, index=top_docs, columns=codes)

# Count occurrences of each code in each document
for doc_id in top_docs:
    doc_data = df[df['document-id'] == doc_id]
    # For rows with code, split and count each code
    for _, row in doc_data.iterrows():
        if isinstance(row['code'], str) and row['code'].strip():
            for code in row['code'].split(','):
                doc_code_counts.loc[doc_id, code.strip()] += 1

print(doc_code_counts)

# Create visualizations
print("\nGenerating visualizations...")

# 1. Stage vs. Code heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(stage_code_counts, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Frequency of Codes by Stage")
plt.tight_layout()
plt.savefig("stage_vs_code_heatmap.png")

# 2. Region vs. Code heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(region_code_counts, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Frequency of Codes by Region")
plt.tight_layout()
plt.savefig("region_vs_code_heatmap.png")

# 3. Document-ID vs. Code heatmap (top documents)
plt.figure(figsize=(14, 10))
sns.heatmap(doc_code_counts, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Frequency of Codes by Document ID (Top 15 Documents)")
plt.tight_layout()
plt.savefig("document_vs_code_heatmap.png")

print("\nAnalysis complete! Tables and heatmaps have been generated.") 