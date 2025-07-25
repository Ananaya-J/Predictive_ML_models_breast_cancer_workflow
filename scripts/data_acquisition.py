# Phase 1: Data Loading and Preprocessing
# GSE20685 Breast Cancer Gene Expression Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING FROM LOCAL SERIES MATRIX FILE
# =============================================================================

def load_gse20685_from_file(file_path):
    """
    Load GSE20685 data from downloaded series matrix file
    
    Parameters:
    file_path (str): Path to the GSE20685_series_matrix.txt file
    
    Returns:
    expression_df (DataFrame): Gene expression data (samples x genes)
    clinical_df (DataFrame): Clinical metadata
    """
    print(f"Loading data from: {file_path}")
    
    # Read the series matrix file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the data section
    data_start = None
    data_end = None
    sample_info = {}
    
    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            data_start = i + 1
        elif line.startswith("!series_matrix_table_end"):
            data_end = i
            break
        elif line.startswith("!Sample_title"):
            titles = line.strip().split('\t')[1:]
            sample_info['titles'] = titles
        elif line.startswith("!Sample_geo_accession"):
            accessions = line.strip().split('\t')[1:]
            sample_info['accessions'] = accessions
        elif line.startswith("!Sample_characteristics_ch1"):
            if 'characteristics' not in sample_info:
                sample_info['characteristics'] = []
            chars = line.strip().split('\t')[1:]
            sample_info['characteristics'].append(chars)
    
    # Extract expression data
    data_lines = lines[data_start:data_end]
    
    # Parse expression data
    expression_data = []
    gene_ids = []
    
    for line in data_lines:
        parts = line.strip().split('\t')
        
        # Skip header row (usually starts with ID_REF)
        if parts[0].upper() == "ID_REF":
            continue

        gene_id = parts[0].strip('"')
        try:
            expression_values = [
                float(x.strip('"')) if x.lower() != 'null' else np.nan
                for x in parts[1:]
            ]
        except ValueError:
            print(f"Skipping line due to ValueError: {line.strip()}")
            continue

        gene_ids.append(gene_id)
        expression_data.append(expression_values)

    
    # Create expression DataFrame
    expression_df = pd.DataFrame(expression_data, 
                                index=gene_ids, 
                                columns=sample_info['accessions'])
    
    # Transpose so samples are rows and genes are columns
    expression_df = expression_df.T
    
    # Create clinical DataFrame from sample information
    clinical_df = pd.DataFrame(index=sample_info['accessions'])
    clinical_df['sample_title'] = sample_info['titles']
    
    # Parse characteristics if available
    if 'characteristics' in sample_info:
        for i, char_line in enumerate(sample_info['characteristics']):
            char_name = f'characteristic_{i+1}'
            clinical_df[char_name] = char_line
    
    print(f"Successfully loaded:")
    print(f"- Expression data: {expression_df.shape[0]} samples x {expression_df.shape[1]} genes")
    print(f"- Clinical data: {clinical_df.shape[0]} samples x {clinical_df.shape[1]} variables")
    
    return expression_df, clinical_df

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

def perform_comprehensive_eda(expression_df, clinical_df):
    """Perform comprehensive exploratory data analysis"""
    
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"\nDataset Overview:")
    print(f"- Number of samples: {expression_df.shape[0]}")
    print(f"- Number of genes/probes: {expression_df.shape[1]}")
    print(f"- Clinical variables: {clinical_df.shape[1]}")
    
    # Missing values analysis
    expr_missing = expression_df.isnull().sum().sum()
    clinical_missing = clinical_df.isnull().sum().sum()
    print(f"\nMissing Values:")
    print(f"- Expression data: {expr_missing} missing values ({expr_missing/(expression_df.shape[0]*expression_df.shape[1])*100:.2f}%)")
    print(f"- Clinical data: {clinical_missing} missing values")
    
    # Expression data statistics
    print(f"\nExpression Data Statistics:")
    print(f"- Min value: {expression_df.min().min():.2f}")
    print(f"- Max value: {expression_df.max().max():.2f}")
    print(f"- Mean expression: {expression_df.mean().mean():.2f}")
    print(f"- Std expression: {expression_df.std().mean():.2f}")
    
    # Create visualization plots
    plt.figure(figsize=(20, 15))
    
    # 1. Expression distribution
    plt.subplot(3, 4, 1)
    sample_data = expression_df.values.flatten()
    sample_data = sample_data[~np.isnan(sample_data)][:10000]  # Sample for plotting
    plt.hist(sample_data, bins=50, alpha=0.7, color='skyblue')
    plt.title('Expression Values Distribution')
    plt.xlabel('Expression Level')
    plt.ylabel('Frequency')
    
    # 2. Sample-wise mean expression
    plt.subplot(3, 4, 2)
    sample_means = expression_df.mean(axis=1)
    plt.hist(sample_means, bins=30, alpha=0.7, color='orange')
    plt.title('Sample-wise Mean Expression')
    plt.xlabel('Mean Expression')
    plt.ylabel('Number of Samples')
    
    # 3. Gene-wise variance
    plt.subplot(3, 4, 3)
    gene_vars = expression_df.var(axis=0)
    plt.hist(gene_vars, bins=50, alpha=0.7, color='green')
    plt.title('Gene-wise Variance Distribution')
    plt.xlabel('Variance')
    plt.ylabel('Number of Genes')
    plt.yscale('log')
    
    # 4. Missing values heatmap
    plt.subplot(3, 4, 4)
    missing_pattern = expression_df.isnull().sum(axis=1)
    plt.hist(missing_pattern, bins=20, alpha=0.7, color='red')
    plt.title('Missing Values per Sample')
    plt.xlabel('Number of Missing Genes')
    plt.ylabel('Number of Samples')
    
    # 5. Sample correlation (first 50 samples)
    plt.subplot(3, 4, 5)
    sample_subset = expression_df.iloc[:50, :100].T
    corr_matrix = sample_subset.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.5})
    plt.title('Sample Correlation (First 50)')
    
    # 6. Top variable genes correlation
    plt.subplot(3, 4, 6)
    top_var_genes = gene_vars.nlargest(50).index
    gene_corr = expression_df[top_var_genes].corr()
    sns.heatmap(gene_corr, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.5})
    plt.title('Top 50 Variable Genes Correlation')
    
    # 7. Boxplot of expression across samples (subset)
    plt.subplot(3, 4, 7)
    sample_subset_expr = expression_df.iloc[:20, :].T
    sample_subset_expr.boxplot(ax=plt.gca(), rot=90)
    plt.title('Expression Distribution (First 20 Samples)')
    plt.ylabel('Expression Level')
    
    # 8. Gene expression range
    plt.subplot(3, 4, 8)
    gene_ranges = expression_df.max(axis=0) - expression_df.min(axis=0)
    plt.hist(gene_ranges, bins=50, alpha=0.7, color='purple')
    plt.title('Gene Expression Range Distribution')
    plt.xlabel('Expression Range')
    plt.ylabel('Number of Genes')
    
    # 9. Mean vs Variance plot
    plt.subplot(3, 4, 9)
    gene_means = expression_df.mean(axis=0)
    plt.scatter(gene_means, gene_vars, alpha=0.5, s=1)
    plt.xlabel('Gene Mean Expression')
    plt.ylabel('Gene Variance')
    plt.title('Mean vs Variance Plot')
    plt.xscale('log')
    plt.yscale('log')
    
    # 10. Sample clustering dendrogram (subset)
    plt.subplot(3, 4, 10)
    from scipy.cluster.hierarchy import dendrogram, linkage
    sample_subset = expression_df.iloc[:30, :100]
    linkage_matrix = linkage(sample_subset, method='ward')
    dendrogram(linkage_matrix, ax=plt.gca(), leaf_rotation=90)
    plt.title('Sample Clustering (First 30)')
    
    # 11. Clinical data overview
    plt.subplot(3, 4, 11)
    clinical_summary = pd.DataFrame({
        'Variable': clinical_df.columns,
        'Non_null_count': clinical_df.notna().sum(),
        'Unique_values': clinical_df.nunique()
    })
    
    plt.text(0.1, 0.9, 'Clinical Variables:', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
    for i, (_, row) in enumerate(clinical_summary.iterrows()):
        plt.text(0.1, 0.8-i*0.1, f"{row['Variable']}: {row['Non_null_count']} non-null", 
                transform=plt.gca().transAxes, fontsize=10)
    plt.axis('off')
    plt.title('Clinical Data Summary')
    
    # 12. Quality metrics
    plt.subplot(3, 4, 12)
    quality_metrics = {
        'Samples': expression_df.shape[0],
        'Genes': expression_df.shape[1],
        'Complete_cases': expression_df.dropna().shape[0],
        'High_var_genes': (gene_vars > gene_vars.quantile(0.75)).sum()
    }
    
    plt.bar(quality_metrics.keys(), quality_metrics.values(), color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    plt.title('Dataset Quality Metrics')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return gene_vars

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================

def preprocess_expression_data(expression_df, clinical_df, n_features=1000):
    """
    Comprehensive preprocessing pipeline for gene expression data
    
    Parameters:
    expression_df (DataFrame): Raw expression data
    clinical_df (DataFrame): Clinical metadata
    n_features (int): Number of top variable features to select
    
    Returns:
    dict: Preprocessed data dictionary
    """
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # 1. Handle missing values
    print("1. Handling missing values...")
    initial_missing = expression_df.isnull().sum().sum()
    expression_df_filled = expression_df.fillna(expression_df.median())
    print(f"   - Filled {initial_missing} missing values with median")
    
    # 2. Check if data needs log transformation
    print("2. Checking data transformation...")
    max_val = expression_df_filled.max().max()
    if max_val > 100:
        print(f"   - Max value: {max_val:.2f}, applying log2 transformation...")
        expression_df_log = np.log2(expression_df_filled + 1)
    else:
        print(f"   - Max value: {max_val:.2f}, data appears already log-transformed")
        expression_df_log = expression_df_filled
    
    # 3. Variance filtering
    print("3. Applying variance filtering...")
    gene_vars = expression_df_log.var(axis=0)
    var_threshold = gene_vars.quantile(0.1)  # Remove bottom 10%
    high_var_genes = gene_vars[gene_vars > var_threshold].index
    expression_df_filtered = expression_df_log[high_var_genes]
    
    removed_genes = len(gene_vars) - len(high_var_genes)
    print(f"   - Removed {removed_genes} low-variance genes")
    print(f"   - Remaining genes: {len(high_var_genes)}")
    
    # 4. Feature selection - top K most variable genes
    print(f"4. Selecting top {n_features} most variable genes...")
    top_var_genes = gene_vars.nlargest(n_features).index
    expression_final = expression_df_log[top_var_genes]
    print(f"   - Selected {len(top_var_genes)} features")
    
    # 5. Standardization
    print("5. Standardizing features...")
    scaler = StandardScaler()
    expression_scaled = pd.DataFrame(
        scaler.fit_transform(expression_final),
        index=expression_final.index,
        columns=expression_final.columns
    )
    
    # 6. Create target variable (for demonstration - modify based on actual clinical data)
    print("6. Preparing target variable...")
    
    # Try to extract molecular subtypes from clinical data
    target_column = None
    target_names = None
    y = None
    label_encoder = None
    
    # Look for relevant clinical variables
    for col in clinical_df.columns:
        col_lower = col.lower()
        unique_vals = clinical_df[col].nunique()
        
        if any(keyword in col_lower for keyword in ['subtype', 'class', 'group', 'type']) and 2 <= unique_vals <= 6:
            target_column = col
            break
    
    if target_column:
        print(f"   - Found target variable: {target_column}")
        # Clean the target variable
        target_data = clinical_df[target_column].dropna()
        
        # Encode categorical target
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(target_data)
        target_names = label_encoder.classes_
        
        # Align expression data with available targets
        common_samples = expression_scaled.index.intersection(target_data.index)
        expression_scaled = expression_scaled.loc[common_samples]
        y = pd.Series(y, index=target_data.index).loc[common_samples].values
        
        print(f"   - Target classes: {target_names}")
        print(f"   - Class distribution: {np.bincount(y)}")
        
    else:
        print("   - No suitable target variable found in clinical data")
        print("   - Creating binary target based on expression patterns for demonstration")
        
        # Create binary target based on first principal component or clustering
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(expression_scaled)
        y = (pc1.flatten() > np.median(pc1)).astype(int)
        target_names = ['Low_PC1', 'High_PC1']
        
        print(f"   - Created binary target based on PC1")
        print(f"   - Class distribution: {np.bincount(y)}")
    
    # 7. Train-test split
    print("7. Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        expression_scaled, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"   - Train set: {X_train.shape[0]} samples")
    print(f"   - Test set: {X_test.shape[0]} samples")
    
    # Create results dictionary
    results = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'target_names': target_names,
        'feature_names': expression_scaled.columns.tolist(),
        'gene_variances': gene_vars,
        'target_column': target_column,
        'expression_processed': expression_scaled,
        'clinical_data': clinical_df
    }
    
    print("\nPreprocessing completed successfully!")
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Specify the path to your downloaded series matrix file
    FILE_PATH = "GSE20685_series_matrix.txt"  # Update this path
    
    try:
        # Load data
        print("Starting Phase 1: Data Loading and Preprocessing")
        expression_df, clinical_df = load_gse20685_from_file(FILE_PATH)
        
        # Perform EDA
        gene_variances = perform_comprehensive_eda(expression_df, clinical_df)
        
        # Preprocess data
        preprocessed_data = preprocess_expression_data(expression_df, clinical_df, n_features=1000)
        
        # Save preprocessed data for next phases
        import pickle
        with open('preprocessed_data_phase1.pkl', 'wb') as f:
            pickle.dump(preprocessed_data, f)
        
        print("\nPhase 1 completed successfully!")
        print("Preprocessed data saved as 'preprocessed_data_phase1.pkl'")
        print("\nNext: Run Phase 2 for model development")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{FILE_PATH}'")
        print("Please update the FILE_PATH variable with the correct path to your series matrix file")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your data file and try again")