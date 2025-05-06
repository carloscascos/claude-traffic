#!/usr/bin/env python3
"""
Parameter Tuning Module

This module adjusts clustering parameters for optimal route detection,
calibrates filtering thresholds for noise reduction, and optimizes
visualization settings.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def load_data(input_file):
    """
    Load processed vessel data from CSV file.
    
    Args:
        input_file (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with vessel positions
    """
    print(f"Loading data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records.")
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def tune_clustering_parameters(df, min_eps=0.005, max_eps=0.05, steps=10, 
                              min_samples_range=[2, 3, 5, 7, 10]):
    """
    Tune DBSCAN clustering parameters for optimal route detection.
    
    Args:
        df (pandas.DataFrame): DataFrame with vessel positions
        min_eps (float): Minimum eps value to test
        max_eps (float): Maximum eps value to test
        steps (int): Number of eps values to test
        min_samples_range (list): List of min_samples values to test
        
    Returns:
        dict: Dictionary with optimal parameters
    """
    print("Tuning clustering parameters...")
    
    if df.empty:
        print("No data to tune. Exiting.")
        return {'eps': 0.02, 'min_samples': 3}
    
    # Check if we have the necessary columns
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("Missing required columns (latitude, longitude).")
        return {'eps': 0.02, 'min_samples': 3}
    
    # Extract coordinates for clustering
    coordinates = df[['longitude', 'latitude']].values
    
    # Prepare eps values to test
    eps_values = np.linspace(min_eps, max_eps, steps)
    
    # Store results
    results = []
    
    # Test different parameter combinations
    for eps in eps_values:
        for min_samples in min_samples_range:
            try:
                # Apply DBSCAN clustering
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
                
                # Get the number of clusters (excluding noise)
                labels = clustering.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Calculate the percentage of non-noise points
                non_noise_percentage = np.sum(labels != -1) / len(labels) * 100
                
                # Calculate silhouette score if we have more than one cluster
                if n_clusters > 1 and non_noise_percentage > 10:
                    # Only calculate silhouette for non-noise points
                    non_noise_mask = (labels != -1)
                    if np.sum(non_noise_mask) > 1:
                        silhouette = silhouette_score(
                            coordinates[non_noise_mask], 
                            labels[non_noise_mask]
                        )
                    else:
                        silhouette = 0
                else:
                    silhouette = 0
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'non_noise_percentage': non_noise_percentage,
                    'silhouette': silhouette,
                    'score': silhouette * (non_noise_percentage / 100) * (n_clusters / len(eps_values))
                })
                
                print(f"  eps={eps:.4f}, min_samples={min_samples}: {n_clusters} clusters, "
                      f"{non_noise_percentage:.1f}% non-noise, silhouette={silhouette:.3f}")
                
            except Exception as e:
                print(f"Error with eps={eps}, min_samples={min_samples}: {e}")
    
    if not results:
        print("No valid parameter combinations found.")
        return {'eps': 0.02, 'min_samples': 3}
    
    # Find the best parameters
    results_df = pd.DataFrame(results)
    best_params = results_df.loc[results_df['score'].idxmax()]
    
    print(f"\nOptimal parameters: eps={best_params['eps']:.4f}, "
          f"min_samples={best_params['min_samples']}")
    print(f"  Resulting in {best_params['n_clusters']} clusters, "
          f"{best_params['non_noise_percentage']:.1f}% non-noise points")
    
    return {
        'eps': float(best_params['eps']),
        'min_samples': int(best_params['min_samples'])
    }

def determine_fleet_colors():
    """
    Determine colors for different fleet types.
    
    Returns:
        dict: Dictionary mapping fleet types to colors
    """
    # Default fleet colors
    fleet_colors = {
        'Cruise/Passenger': '#E41A1C',  # Red
        'Container': '#377EB8',         # Blue
        'Bulk Carrier': '#4DAF4A',      # Green
        'Tanker': '#984EA3',            # Purple
        'Ferry/Ro-Ro': '#FF7F00',       # Orange
        'Fishing': '#FFFF33',           # Yellow
        'Pleasure': '#A65628',          # Brown
        'Tug': '#F781BF',               # Pink
        'Other': '#999999',             # Grey
        'Unknown': '#000000'            # Black
    }
    
    return fleet_colors

def tune_visualization_settings(df):
    """
    Optimize visualization settings based on the data.
    
    Args:
        df (pandas.DataFrame): DataFrame with vessel positions
        
    Returns:
        dict: Dictionary with visualization settings
    """
    print("Tuning visualization settings...")
    
    # Default settings
    settings = {
        'zoom_level': 8,
        'line_opacity': 0.7,
        'min_line_width': 2,
        'max_line_width': 10,
        'use_log_scale': True,
        'cluster_routes': True,
        'fleet_colors': determine_fleet_colors()
    }
    
    if df.empty:
        return settings
    
    # Try to determine appropriate zoom level based on data spread
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat_range = df['latitude'].max() - df['latitude'].min()
        lon_range = df['longitude'].max() - df['longitude'].min()
        max_range = max(lat_range, lon_range)
        
        # Adjust zoom level based on data spread
        if max_range > 10:
            settings['zoom_level'] = 5
        elif max_range > 5:
            settings['zoom_level'] = 6
        elif max_range > 2:
            settings['zoom_level'] = 7
        elif max_range > 1:
            settings['zoom_level'] = 8
        elif max_range > 0.5:
            settings['zoom_level'] = 9
        else:
            settings['zoom_level'] = 10
    
    # Check if we should use fleet-specific settings
    if 'fleet' in df.columns:
        fleet_counts = df['fleet'].value_counts()
        
        # If we have multiple fleets, ensure we use fleet filtering
        if len(fleet_counts) > 1:
            settings['fleet_filtering'] = True
            
            # Adjust fleet colors based on what's in the data
            fleet_colors = determine_fleet_colors()
            settings['fleet_colors'] = {
                fleet: fleet_colors.get(fleet, '#000000')
                for fleet in fleet_counts.index
            }
        else:
            settings['fleet_filtering'] = False
    else:
        settings['fleet_filtering'] = False
    
    # Set line width scaling based on data
    if 'GT' in df.columns:
        min_gt = df['GT'].min()
        max_gt = df['GT'].max()
        
        # If there's a large range of gross tonnage, use logarithmic scaling
        if max_gt / min_gt > 10:
            settings['use_log_scale'] = True
        else:
            settings['use_log_scale'] = False
    
    return settings

def save_parameters(parameters, output_file):
    """
    Save parameters to a JSON file.
    
    Args:
        parameters (dict): Dictionary with parameters
        output_file (str): Path to save the JSON file
    """
    # Create output directory if it doesn't exist
    directory = os.path.dirname(output_file)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)
    
    # Convert any NumPy or pandas types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list)):
            return [convert_to_native(x) for x in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        else:
            return obj
    
    # Convert parameters to native Python types
    native_parameters = convert_to_native(parameters)
    
    print(f"Saving parameters to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(native_parameters, f, indent=2)
    print(f"Parameters saved successfully to {output_file}")

def main():
    """
    Main function to tune parameters for route analysis.
    """
    parser = argparse.ArgumentParser(description='Tune parameters for route analysis.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with processed data')
    parser.add_argument('--output', type=str, default='data/tuned_params.json', help='Output JSON file for parameters')
    
    args = parser.parse_args()
    
    # Print the parameters
    print(f"Tuning parameters with inputs:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    
    # Load the data
    df = load_data(args.input)
    
    if df.empty:
        print("No data to process. Exiting.")
        return
    
    # Tune parameters
    clustering_params = tune_clustering_parameters(df)
    visualization_settings = tune_visualization_settings(df)
    
    # Combine all parameters
    parameters = {
        'clustering': clustering_params,
        'visualization': visualization_settings
    }
    
    # Save parameters
    save_parameters(parameters, args.output)
    
    print("Parameter tuning completed.")

if __name__ == "__main__":
    main()
