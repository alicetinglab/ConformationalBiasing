#!/usr/bin/env python3
"""
create_complete_enhanced_weighted_averages.py

Creates enhanced weighted averages from Labeled_oligomer_output_csvs_enhanced data
that includes UV260 (avg_uv2) and UV260/280 ratios in addition to the existing UV280 data.

This script is updated to handle any number of volume fractions dynamically, not limited to 3.

This script:
1. Dynamically detects all volume fraction columns in the data
2. Groups data by sample identifier (extracted from filename)
3. Calculates weighted averages of volume fractions using UV280 as weights
4. Calculates total UV260 and UV260/280 ratios
5. Applies the same filtering criteria as the original processing
6. Outputs enhanced weighted average files with UV260 data to weighted_average_UV_complete

Usage:
    python create_complete_enhanced_weighted_averages.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import re

def extract_sample_identifier(filename: str) -> str:
    """
    Extract sample identifier from filename.
    Pattern: YYYYMMDD_Ave_SSample_volume_condition_0_WindowRange_trimmed_0.4.log
    Returns: Everything before the window range
    """
    match = re.match(r'(.+)_\d+-\d+_trimmed_0\.4\.log', filename)
    return match.group(1) if match else filename

def find_volume_fraction_columns(df: pd.DataFrame) -> List[str]:
    """
    Dynamically find all volume fraction columns in the DataFrame.
    Looks for columns matching pattern 'volume_fraction_N' where N is any number.
    """
    volume_fraction_cols = []
    
    # Look for volume_fraction_N pattern
    for col in df.columns:
        if re.match(r'volume_fraction_\d+', col):
            volume_fraction_cols.append(col)
    
    # Sort by the number in the column name
    def extract_number(col_name):
        match = re.search(r'volume_fraction_(\d+)', col_name)
        return int(match.group(1)) if match else 0
    
    volume_fraction_cols.sort(key=extract_number)
    
    return volume_fraction_cols

def calculate_complete_enhanced_weighted_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate enhanced weighted averages including UV260 and UV260/280 ratios.
    This version dynamically handles any number of volume fractions.
    
    Args:
        df: DataFrame with enhanced labeled oligomer data
    
    Returns:
        DataFrame with weighted averages including UV260 data
    """
    # Find all volume fraction columns dynamically
    volume_fraction_cols = find_volume_fraction_columns(df)
    
    if not volume_fraction_cols:
        raise ValueError("No volume fraction columns found in the data!")
    
    print(f"Found {len(volume_fraction_cols)} volume fraction columns: {volume_fraction_cols}")
    
    # Add sample identifier column first
    df['sample_identifier'] = df['filename'].apply(extract_sample_identifier)
    print(f"Processing {len(df)} measurements for {df['sample_identifier'].nunique()} unique samples")
    
    def process_sample(group):
        # Get UV values and fractions
        uv280_values = group['avg_uv1'].values  # UV280
        uv260_values = group['avg_uv2'].values  # UV260
        chi2_values = group['chi2'].values
        
        # Get all volume fraction values dynamically
        frac_values_dict = {}
        for col in volume_fraction_cols:
            frac_values_dict[col] = group[col].values
        
        # Handle NaN values - need at least UV280 for weighted averaging
        valid_mask = ~np.isnan(uv280_values)
        for col in volume_fraction_cols:
            valid_mask = valid_mask & ~np.isnan(frac_values_dict[col])
        
        if not valid_mask.any():
            # Create result with NaN values for all volume fractions
            result = {
                'filename': group['filename'].iloc[0],
                'average_chi2': np.nan,
                'folder': group['Folder:'].iloc[0] if len(group) > 0 else np.nan,
                'subfolder': group['Subfolder'].iloc[0] if len(group) > 0 else np.nan,
                'protein': group['Protein'].iloc[0] if len(group) > 0 else np.nan,
                'condition': group['Condition'].iloc[0] if len(group) > 0 else np.nan,
                'identity': group['identity'].iloc[0] if len(group) > 0 else np.nan,
                'average_uv280': 0,
                'average_uv260': 0,
                'average_uv260_280_ratio': np.nan,
                'total_window_range': '',
                'num_volume_fractions': len(volume_fraction_cols)
            }
            
            # Add NaN values for all volume fractions
            for col in volume_fraction_cols:
                result[col] = np.nan
                
            return pd.Series(result)
        
        uv280_valid = uv280_values[valid_mask]
        uv260_valid = uv260_values[valid_mask]
        chi2_valid = chi2_values[valid_mask]
        
        # Get valid fraction values for all volume fractions
        frac_valid_dict = {}
        for col in volume_fraction_cols:
            frac_valid_dict[col] = frac_values_dict[col][valid_mask]
        
        # Calculate weighted averages using UV280 as weights
        total_uv280 = np.sum(uv280_valid)
        total_uv260 = np.sum(uv260_valid)
        avg_chi2 = np.mean(chi2_valid)
        
        # Calculate UV260/280 ratio
        if total_uv280 > 0:
            uv260_280_ratio = total_uv260 / total_uv280
        else:
            uv260_280_ratio = np.nan
        
        # Get all window ranges for this sample
        window_ranges = group['window_range'].dropna().unique()
        total_window_range = ', '.join(sorted(window_ranges))
        
        # Calculate weighted averages for all volume fractions
        weighted_fracs = {}
        if total_uv280 == 0:
            # If no UV280 signal, use simple average
            for col in volume_fraction_cols:
                weighted_fracs[col] = np.mean(frac_valid_dict[col])
        else:
            # Weighted average: sum(frac * UV280) / sum(UV280)
            for col in volume_fraction_cols:
                weighted_fracs[col] = np.sum(frac_valid_dict[col] * uv280_valid) / total_uv280
        
        # Create result dictionary
        result = {
            'filename': group['filename'].iloc[0],
            'average_chi2': avg_chi2,
            'folder': group['Folder:'].iloc[0] if len(group) > 0 else np.nan,
            'subfolder': group['Subfolder'].iloc[0] if len(group) > 0 else np.nan,
            'protein': group['Protein'].iloc[0] if len(group) > 0 else np.nan,
            'condition': group['Condition'].iloc[0] if len(group) > 0 else np.nan,
            'identity': group['identity'].iloc[0] if len(group) > 0 else np.nan,
            'average_uv280': total_uv280,
            'average_uv260': total_uv260,
            'average_uv260_280_ratio': uv260_280_ratio,
            'total_window_range': total_window_range,
            'num_volume_fractions': len(volume_fraction_cols)
        }
        
        # Add weighted averages for all volume fractions
        result.update(weighted_fracs)
        
        return pd.Series(result)
    
    # Group by sample_identifier and apply weighted average calculation
    weighted_df = df.groupby('sample_identifier').apply(process_sample).reset_index(drop=True)
    
    # Apply filtering criteria similar to original processing
    print(f"Before filtering: {len(weighted_df)} samples")
    
    # Remove samples with missing protein/condition information
    if 'protein' in weighted_df.columns:
        weighted_df = weighted_df[weighted_df['protein'].notna()]
    if 'condition' in weighted_df.columns:
        weighted_df = weighted_df[weighted_df['condition'].notna()]
    
    # Remove samples with very low UV280 signals (likely noise)
    weighted_df = weighted_df[weighted_df['average_uv280'] > 10]
    
    # Remove samples with poor chi2 values
    weighted_df = weighted_df[weighted_df['average_chi2'] < 10]
    
    print(f"After filtering: {len(weighted_df)} samples")
    
    return weighted_df

def process_single_pdb_set(csv_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Process a single enhanced CSV file and create enhanced weighted averages."""
    print(f"Processing {csv_path.name}...")
    
    # Read the enhanced CSV file
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['filename', 'avg_uv1', 'avg_uv2', 'chi2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Missing columns {missing_cols} in {csv_path.name}")
        return {'file': csv_path.name, 'error': f"Missing columns: {missing_cols}"}
    
    # Find volume fraction columns dynamically
    volume_fraction_cols = find_volume_fraction_columns(df)
    if not volume_fraction_cols:
        print(f"  Warning: No volume fraction columns found in {csv_path.name}")
        return {'file': csv_path.name, 'error': "No volume fraction columns found"}
    
    print(f"  Found {len(volume_fraction_cols)} volume fraction columns: {volume_fraction_cols}")
    
    # Calculate enhanced weighted averages
    weighted_df = calculate_complete_enhanced_weighted_averages(df)
    
    # Create output filename
    output_filename = csv_path.name.replace('_oligomer_data_labeled.csv', 
                                          '_oligomer_data_labeled_complete_weighted_average.csv')
    output_path = output_dir / output_filename
    
    # Save enhanced weighted averages
    weighted_df.to_csv(output_path, index=False)
    
    # Calculate statistics
    total_samples = len(weighted_df)
    valid_samples = weighted_df[volume_fraction_cols[0]].notna().sum()
    
    # Calculate average fractions for all volume fractions
    avg_fracs = {}
    for col in volume_fraction_cols:
        avg_fracs[f'avg_{col}'] = weighted_df[col].mean()
    
    result = {
        'file': csv_path.name,
        'output_file': output_path,
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'num_volume_fractions': len(volume_fraction_cols),
        'volume_fraction_columns': volume_fraction_cols,
        'avg_uv280': weighted_df['average_uv280'].mean(),
        'avg_uv260': weighted_df['average_uv260'].mean(),
        'avg_uv260_280_ratio': weighted_df['average_uv260_280_ratio'].mean()
    }
    
    # Add average fractions
    result.update(avg_fracs)
    
    return result

def process_all_enhanced_files(input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Process all enhanced CSV files."""
    # Find all enhanced CSV files
    csv_files = list(input_dir.glob("*_oligomer_data_labeled.csv"))
    
    if not csv_files:
        print("No enhanced labeled oligomer CSV files found!")
        return []
    
    print(f"Found {len(csv_files)} enhanced CSV files")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for csv_file in sorted(csv_files):
        result = process_single_pdb_set(csv_file, output_dir)
        results.append(result)
        
        if 'error' in result:
            print(f"  {result['file']}: ERROR - {result['error']}")
        else:
            print(f"  {result['file']}: {result['valid_samples']}/{result['total_samples']} samples")
            print(f"    Volume fractions: {result['num_volume_fractions']} ({', '.join(result['volume_fraction_columns'])})")
            
            # Print average fractions for all volume fractions
            frac_summary = []
            for col in result['volume_fraction_columns']:
                avg_key = f'avg_{col}'
                if avg_key in result:
                    frac_summary.append(f"{col}={result[avg_key]:.3f}")
            print(f"    Avg fractions: {', '.join(frac_summary)}")
            print(f"    Avg UV280: {result['avg_uv280']:.1f}, Avg UV260: {result['avg_uv260']:.1f}")
            print(f"    Avg UV260/280 ratio: {result['avg_uv260_280_ratio']:.3f}")
    
    return results

def create_combined_complete_summary(results: List[Dict[str, Any]], output_dir: Path):
    """Create a combined summary of all enhanced weighted averages."""
    print(f"\nCreating combined complete summary...")
    
    # Collect all enhanced weighted average data
    all_data = []
    for result in results:
        if 'error' not in result and result['output_file'].exists():
            df = pd.read_csv(result['output_file'])
            df['pdb_set'] = result['file'].replace('_oligomer_data_labeled.csv', '')
            all_data.append(df)
    
    if not all_data:
        print("No valid data found for combined summary")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True, sort=False)
    
    # Save combined data
    combined_path = output_dir / 'all_pdb_sets_complete_weighted_averages.csv'
    combined_df.to_csv(combined_path, index=False)
    
    # Find all volume fraction columns in the combined data
    volume_fraction_cols = find_volume_fraction_columns(combined_df)
    
    # Create summary statistics
    summary_stats = {
        'total_samples': len(combined_df),
        'unique_proteins': combined_df['protein'].nunique(),
        'unique_conditions': combined_df['condition'].nunique(),
        'num_volume_fractions': len(volume_fraction_cols),
        'avg_uv280': combined_df['average_uv280'].mean(),
        'avg_uv260': combined_df['average_uv260'].mean(),
        'avg_uv260_280_ratio': combined_df['average_uv260_280_ratio'].mean()
    }
    
    # Add average fractions for all volume fractions
    for col in volume_fraction_cols:
        summary_stats[f'avg_{col}'] = combined_df[col].mean()
    
    # Create summary report
    report_path = output_dir / 'complete_weighted_averages_report.txt'
    with open(report_path, 'w') as f:
        f.write("COMPLETE ENHANCED WEIGHTED AVERAGE FRACTIONS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Overall Statistics:\n")
        f.write(f"  Total samples: {summary_stats['total_samples']}\n")
        f.write(f"  Unique proteins: {summary_stats['unique_proteins']}\n")
        f.write(f"  Unique conditions: {summary_stats['unique_conditions']}\n")
        f.write(f"  Number of volume fractions: {summary_stats['num_volume_fractions']}\n")
        f.write(f"  Volume fraction columns: {', '.join(volume_fraction_cols)}\n")
        
        # Print average fractions for all volume fractions
        f.write(f"\nAverage weighted fractions:\n")
        for col in volume_fraction_cols:
            f.write(f"  {col}: {summary_stats[f'avg_{col}']:.4f}\n")
        
        f.write(f"\nUV Statistics:\n")
        f.write(f"  Average UV280: {summary_stats['avg_uv280']:.1f}\n")
        f.write(f"  Average UV260: {summary_stats['avg_uv260']:.1f}\n")
        f.write(f"  Average UV260/280 ratio: {summary_stats['avg_uv260_280_ratio']:.4f}\n\n")
        
        # Condition distribution
        f.write("Condition Distribution:\n")
        condition_dist = combined_df['condition'].value_counts()
        for condition, count in condition_dist.items():
            f.write(f"  {condition}: {count} samples\n")
        
        f.write(f"\nProtein Distribution:\n")
        protein_dist = combined_df['protein'].value_counts()
        for protein, count in protein_dist.head(10).items():
            f.write(f"  {protein}: {count} samples\n")
        
        f.write(f"\nPer-PDB-Set Statistics:\n")
        for result in results:
            if 'error' not in result:
                f.write(f"  {result['file']}:\n")
                f.write(f"    Samples: {result['valid_samples']}/{result['total_samples']}\n")
                f.write(f"    Volume fractions: {result['num_volume_fractions']} ({', '.join(result['volume_fraction_columns'])})\n")
                
                # Print average fractions for all volume fractions
                f.write(f"    Avg fractions: ")
                frac_summary = []
                for col in result['volume_fraction_columns']:
                    avg_key = f'avg_{col}'
                    if avg_key in result:
                        frac_summary.append(f"{col}={result[avg_key]:.3f}")
                f.write(f"{', '.join(frac_summary)}\n")
                
                f.write(f"    Avg UV280: {result['avg_uv280']:.1f}, Avg UV260: {result['avg_uv260']:.1f}\n")
                f.write(f"    Avg UV260/280 ratio: {result['avg_uv260_280_ratio']:.3f}\n")
    
    print(f"Combined data saved to: {combined_path}")
    print(f"Summary report saved to: {report_path}")

def main():
    input_dir = Path("Analysis_9-27-25/Labeled_oligomer_output_csvs_enhanced")
    output_dir = Path("Analysis_9-27-25/weighted_average_UV_complete")
    
    # Validate inputs
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    
    # Process all enhanced files
    print(f"Processing complete enhanced weighted averages from: {input_dir}")
    print(f"Output directory: {output_dir}")
    results = process_all_enhanced_files(input_dir, output_dir)
    
    if results:
        create_combined_complete_summary(results, output_dir)
        print(f"\nCompleted! Complete enhanced weighted average data saved to: {output_dir}")
    else:
        print("No files were processed.")

if __name__ == "__main__":
    main()
