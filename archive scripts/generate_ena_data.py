import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
regions = ["guangdong", "hongkong"]
stages = ["stage1", "stage2", "stage3"]
codes = ["ZA", "HJ", "ZC", "GM", "HX", "LX", "JN", "XS", "ZZ"]
num_documents_per_region_stage = 10
min_stanzas = 50
max_stanzas = 300
min_codes = 1
max_codes = 4

# Region abbreviations
region_abbr = {
    "guangdong": "gd",
    "hongkong": "hk"
}

# Define region-stage specific code co-occurrence patterns
# For each region-stage, we define groups of codes that are more likely to co-occur
co_occurrence_patterns = {
    # Guangdong patterns
    ("guangdong", "stage1"): [["ZA", "HJ", "ZC"], ["GM", "HX"], ["LX", "JN", "XS", "ZZ"]],
    ("guangdong", "stage2"): [["ZA", "ZC"], ["GM", "HX", "LX"], ["JN", "XS", "ZZ", "HJ"]],
    ("guangdong", "stage3"): [["ZA", "GM", "LX"], ["HJ", "HX", "JN"], ["ZC", "XS", "ZZ"]],
    
    # Hongkong patterns
    ("hongkong", "stage1"): [["ZA", "GM", "JN"], ["HJ", "HX", "XS"], ["ZC", "LX", "ZZ"]],
    ("hongkong", "stage2"): [["ZA", "LX", "ZZ"], ["HJ", "ZC", "JN"], ["GM", "HX", "XS"]],
    ("hongkong", "stage3"): [["ZA", "XS"], ["HJ", "ZC", "HX"], ["GM", "LX", "JN", "ZZ"]]
}

# Create empty dataframe to store all data
all_data = []

# Generate data
for region in regions:
    for stage in stages:
        for doc_num in range(1, num_documents_per_region_stage + 1):
            region_prefix = region_abbr[region]
            document_id = f"{region_prefix}_meeting{doc_num}"
            num_stanzas = random.randint(min_stanzas, max_stanzas)
            
            for stanza_id_num in range(1, num_stanzas + 1):
                # Create formatted stanza ID
                stanza_id = f"{region_prefix}_mt{doc_num}_st{stanza_id_num}"
                
                # Get the co-occurrence pattern for this region-stage
                pattern_groups = co_occurrence_patterns[(region, stage)]
                
                # 70% of the time use the region-stage pattern, 30% random
                if random.random() < 0.7:
                    # Choose one of the pattern groups with higher probability for first group
                    group_weights = [0.5, 0.3, 0.2]  # Adjust these weights as needed
                    chosen_group = random.choices(
                        pattern_groups, 
                        weights=group_weights[:len(pattern_groups)], 
                        k=1
                    )[0]
                    
                    # Determine how many codes from the chosen group to use
                    num_codes_to_use = random.randint(
                        1, min(max_codes, len(chosen_group))
                    )
                    
                    # Randomly select codes from the chosen group
                    codes_present = random.sample(chosen_group, num_codes_to_use)
                else:
                    # Random selection as before
                    num_codes_present = random.randint(min_codes, max_codes)
                    codes_present = random.sample(codes, num_codes_present)
                
                # Create row dictionary
                row = {
                    "Region": region,
                    "documentid": document_id,
                    "stanzaid": stanza_id,
                    "stage": stage
                }
                
                # Set code values (0 or 1)
                for code in codes:
                    row[code] = 1 if code in codes_present else 0
                
                all_data.append(row)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
output_file = "ena_dataset.csv"
df.to_csv(output_file, index=False)

print("Dataset generated and saved to", output_file)
print("Total rows:", len(df))
print("Sample of the data:")
print(df.head())

# Provide summary statistics
print("\nSummary statistics:")
print("Number of regions:", df['Region'].nunique())
print("Number of stages:", df['stage'].nunique())
print("Number of documents:", df['documentid'].nunique())
avg_stanzas = df.groupby(['Region', 'documentid', 'stage']).size().mean()
print("Average stanzas per document: {:.2f}".format(avg_stanzas))
print("Code frequency:")
for code in codes:
    frequency = df[code].mean()
    print("  {}: {:.2%}".format(code, frequency))

# Print co-occurrence statistics to verify the patterns
print("\nCode co-occurrence statistics by region and stage:")
for region in regions:
    for stage in stages:
        region_stage_data = df[(df['Region'] == region) & (df['stage'] == stage)]
        print(f"\n{region.capitalize()} - {stage}:")
        
        # Create co-occurrence matrix for this region and stage
        co_occur = np.zeros((len(codes), len(codes)))
        for _, row in region_stage_data.iterrows():
            present_codes = [code for code in codes if row[code] == 1]
            for i, code1 in enumerate(present_codes):
                for j, code2 in enumerate(present_codes):
                    if i != j:
                        i_idx = codes.index(code1)
                        j_idx = codes.index(code2)
                        co_occur[i_idx][j_idx] += 1
        
        # Print top 5 co-occurring code pairs
        code_pairs = []
        for i in range(len(codes)):
            for j in range(i+1, len(codes)):
                pair_count = co_occur[i][j] + co_occur[j][i]
                if pair_count > 0:
                    code_pairs.append((codes[i], codes[j], pair_count))
        
        code_pairs.sort(key=lambda x: x[2], reverse=True)
        for code1, code2, count in code_pairs[:5]:
            print(f"  {code1}-{code2}: {count:.0f}") 