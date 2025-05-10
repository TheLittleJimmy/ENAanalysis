import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Set font that supports Chinese characters if available
font_path = None
chinese_fonts = [
    'Arial Unicode MS', 'SimHei', 'Microsoft YaHei',
    'WenQuanYi Zen Hei', 'Hiragino Sans GB', 'PingFang SC'
]

for font in chinese_fonts:
    try:
        font_path = fm.findfont(fm.FontProperties(family=font))
        if font_path:
            plt.rcParams['font.family'] = font
            break
    except Exception:
        continue

# If no Chinese font found, use a default font and print a warning
if font_path is None:
    print("Warning: No font with Chinese character support found.")
    print("Using default font. Chinese characters may not display correctly.")

# Load the data
df = pd.read_csv('enadata_processed.csv')

# Check the columns
print("Columns:", df.columns.tolist())
print("\nData shape:", df.shape)

# Map Chinese names to English
stage_map = {
    '初期': 'Early',
    '中期': 'Middle',
    '后期': 'Late'
}

region_map = {
    '广东': 'Guangdong',
    '香港': 'Hong Kong',
    '未知地区': 'Unknown Area'
}

# Apply mapping
df['stage_en'] = df['stage'].map(stage_map)
df['region_en'] = df['Region'].map(region_map)

# Get unique codes - these are the actual code columns in the dataset
code_columns = ['AA', 'CG', 'CM', 'IL', 'KB', 'LD', 'LP', 'MT', 'MU', 'PA', 'RI']
print("\nCode columns:", code_columns)

# Get unique stages
stages = sorted(df['stage_en'].unique())
print("\nUnique stages:", stages)

# Get unique regions
regions = sorted(df['region_en'].unique())
print("\nUnique regions:", regions)

# Get unique document IDs
document_ids = df['document-id'].unique()
print("\nNumber of unique document IDs:", len(document_ids))
print("\nUnique document IDs:", document_ids)

# 1. Create a table of stage vs. code
print("\n1. Creating stage vs. code table...")
stage_code_counts = pd.DataFrame(0, index=stages, columns=code_columns)

# Count occurrences of each code in each stage
for stage in stages:
    stage_data = df[df['stage_en'] == stage]
    for code in code_columns:
        stage_code_counts.loc[stage, code] = stage_data[code].sum()

print(stage_code_counts)

# 2. Create a table of region vs. code
print("\n2. Creating region vs. code table...")
region_code_counts = pd.DataFrame(0, index=regions, columns=code_columns)

# Count occurrences of each code in each region
for region in regions:
    region_data = df[df['region_en'] == region]
    for code in code_columns:
        region_code_counts.loc[region, code] = region_data[code].sum()

print(region_code_counts)

# 3. Create a table of document-id vs. code
print("\n3. Creating document-id vs. code table...")
# Since there might be many document IDs, limit to top 10 by number of entries
doc_code_counts = pd.DataFrame(0, index=document_ids, columns=code_columns)

# Count occurrences of each code in each document
for doc_id in document_ids:
    doc_data = df[df['document-id'] == doc_id]
    for code in code_columns:
        doc_code_counts.loc[doc_id, code] = doc_data[code].sum()

print(doc_code_counts)

# Create shorter document labels for better visualization
doc_labels = {}
for i, doc_id in enumerate(document_ids):
    if len(doc_id) > 15:
        doc_labels[doc_id] = f"Doc {i+1}: {doc_id[:15]}..."
    else:
        doc_labels[doc_id] = f"Doc {i+1}: {doc_id}"

doc_code_counts.index = [doc_labels[doc_id] for doc_id in doc_code_counts.index]

# Create visualizations
print("\nGenerating visualizations...")

# Define a color map with better contrast
cmap = sns.color_palette("YlGnBu", as_cmap=True)

# 1. Stage vs. Code heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(stage_code_counts, annot=True, cmap=cmap, fmt="d")
plt.title("Frequency of Codes by Stage", fontsize=16)
plt.xlabel("Code", fontsize=12)
plt.ylabel("Stage", fontsize=12)
plt.tight_layout()
plt.savefig("stage_vs_code_heatmap.png", dpi=300)

# 2. Region vs. Code heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(region_code_counts, annot=True, cmap=cmap, fmt="d")
plt.title("Frequency of Codes by Region", fontsize=16)
plt.xlabel("Code", fontsize=12)
plt.ylabel("Region", fontsize=12)
plt.tight_layout()
plt.savefig("region_vs_code_heatmap.png", dpi=300)

# 3. Document-ID vs. Code heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(doc_code_counts, annot=True, cmap=cmap, fmt="d")
plt.title("Frequency of Codes by Document ID", fontsize=16)
plt.xlabel("Code", fontsize=12)
plt.ylabel("Document ID", fontsize=12)
plt.tight_layout()
plt.savefig("document_vs_code_heatmap.png", dpi=300)

# Save the tables to CSV files for easier reference
stage_code_counts.to_csv('stage_vs_code_table.csv')
region_code_counts.to_csv('region_vs_code_table.csv')
doc_code_counts.to_csv('document_vs_code_table.csv')

print("\nAnalysis complete! Tables and heatmaps have been generated.")
print("CSV files have been saved for each table.") 