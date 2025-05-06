#!/usr/bin/env python3
"""
DBSCAN Parameter Tuning for Maritime Route Visualization

This script analyzes vessel position data and automatically determines
optimal DBSCAN parameters to ensure render.py produces meaningful output.
It uses a binary search approach to efficiently find parameters that produce
at least unclustered individual routes.

Usage:
    python tune.py --input data/vessel_routes.csv [--output data/params.json]
"""

import os
import argparse
import json
import time
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import sys
from datetime import datetime

def find_latest_csv(directory='data'):
    """Find the most recently modified CSV file in the specified directory."""
    csv_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append((full_path, os.path.getmtime(full_path)))
    
    if not csv_files:
        print(f"No CSV files found in {directory} directory.")
        return None
    
    # Sort by modification time (newest first)
    csv_files.sort(key=lambda x: x[1], reverse=True)
    return csv_files[0][0]  # Return the path of the most recently modified file

def load_vessel_data(input_file):
    """Load vessel position data from CSV file."""
    print(f"Loading vessel data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records.")
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def filter_data(df, min_speed, min_trip_points):
    """Filter the data to remove stationary positions and short trips."""
    print(f"Filtering data (min_speed={min_speed}, min_trip_points={min_trip_points})...")
    
    # Filter out positions where the vessel is not moving
    speed_column = 'calculated_speed' if 'calculated_speed' in df.columns else 'speed'
    
    if speed_column in df.columns:
        moving_df = df[df[speed_column] >= min_speed].copy()
        print(f"After speed filtering: {len(moving_df)} records")
    else:
        print("Warning: No speed column found. Cannot filter by speed.")
        moving_df = df.copy()
    
    # Count points per trip
    trip_counts = moving_df.groupby('unique_trip_id').size()
    
    # Filter out trips with too few points
    valid_trips = trip_counts[trip_counts >= min_trip_points].index
    valid_df = moving_df[moving_df['unique_trip_id'].isin(valid_trips)].copy()
    
    print(f"After trip length filtering: {len(valid_df)} records")
    print(f"Valid trips: {len(valid_trips)}")
    
    return valid_df, len(valid_trips)

def create_trip_lines(df):
    """Convert trip points to LineString objects."""
    print("Creating trip lines...")
    
    # Check column names for latitude and longitude
    lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
    lng_col = 'longitude' if 'longitude' in df.columns else 'lng'
    
    if lat_col not in df.columns or lng_col not in df.columns:
        print(f"Error: Required columns not found. Available columns: {df.columns.tolist()}")
        return gpd.GeoDataFrame()
    
    # Group by trip ID and create LineString for each trip
    trip_lines = []
    trip_data = []
    
    for trip_id, trip_points in df.groupby('unique_trip_id'):
        # Sort points by timestamp
        trip_points = trip_points.sort_values('timestamp')
        
        # Create line coordinates
        line_coords = [(row[lng_col], row[lat_col]) for _, row in trip_points.iterrows()]
        
        if len(line_coords) >= 2:
            # Create LineString
            line = LineString(line_coords)
            
            # Add trip data
            imo = trip_points['imo'].iloc[0]
            vessel_name = trip_points['vessel_name'].iloc[0] if 'vessel_name' in trip_points else 'Unknown'
            
            # Get vessel type and fleet if available
            vessel_type = trip_points['vessel_type'].iloc[0] if 'vessel_type' in trip_points else None
            fleet = trip_points['fleet'].iloc[0] if 'fleet' in trip_points else None
            
            # Get gross tonnage if available
            gross_tonnage = trip_points['GT'].iloc[0] if 'GT' in trip_points else 0
            
            # Get timestamps
            start_time = trip_points['timestamp'].min()
            end_time = trip_points['timestamp'].max()
            
            # Calculate duration
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            trip_lines.append(line)
            trip_data.append({
                'trip_id': trip_id,
                'imo': imo,
                'vessel_name': vessel_name,
                'vessel_type': vessel_type,
                'fleet': fleet,
                'gross_tonnage': gross_tonnage,
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': duration_hours,
                'point_count': len(trip_points)
            })
    
    # Create GeoDataFrame with trip lines
    if not trip_lines:
        print("No valid trip lines created.")
        return gpd.GeoDataFrame()
    
    gdf = gpd.GeoDataFrame(trip_data, geometry=trip_lines, crs="EPSG:4326")
    print(f"Created {len(gdf)} trip lines.")
    
    return gdf

def simplify_trip_lines(gdf, tolerance=0.001):
    """Simplify trip lines to reduce complexity."""
    print(f"Simplifying trip lines (tolerance={tolerance})...")
    
    # Create a copy to avoid modifying the original
    simplified_gdf = gdf.copy()
    
    # Simplify geometries
    simplified_gdf['geometry'] = simplified_gdf['geometry'].simplify(tolerance)
    
    return simplified_gdf

def extract_line_features(line, num_points=40):
    """Extract features from a LineString by sampling points along the line."""
    if not line.is_empty:
        try:
            # Sample points along the line
            distances = np.linspace(0, line.length, num_points)
            points = [line.interpolate(distance) for distance in distances]
            
            # Extract coordinates
            features = []
            for point in points:
                features.extend([point.x, point.y])
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    return None

def cluster_routes(gdf, eps, min_samples, by_vessel_type=False):
    """Cluster similar routes using DBSCAN."""
    print(f"Clustering routes (eps={eps}, min_samples={min_samples})...")
    
    if by_vessel_type and 'vessel_type' in gdf.columns:
        # Cluster separately by vessel type (not our focus for tuning)
        raise NotImplementedError("By-vessel-type clustering not implemented for parameter tuning")
    
    # Extract features from each line
    features = []
    valid_indices = []
    
    for idx, row in gdf.iterrows():
        line_features = extract_line_features(row.geometry)
        if line_features:
            features.append(line_features)
            valid_indices.append(idx)
    
    if not features:
        print("No valid features extracted for clustering.")
        return 0, 0  # No clusters, no routes
    
    # Convert to numpy array
    features_array = np.array(features)
    
    # Normalize features
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features_array)
    
    # Count clusters
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_unclustered = (clustering.labels_ == -1).sum()
    
    print(f"Found {n_clusters} route clusters. {n_unclustered} routes were not clustered.")
    
    return n_clusters, len(valid_indices) - n_unclustered

def evaluate_parameters(df, params):
    """
    Evaluate a set of parameters and return success metrics.
    
    Returns:
        tuple: (valid_trips, has_lines, n_clusters, n_clustered_routes)
    """
    # Filter data
    filtered_df, valid_trips = filter_data(df, params['min_speed'], params['min_trip_points'])
    
    if filtered_df.empty:
        return valid_trips, 0, 0, 0
    
    # Create and simplify trip lines
    trip_lines_gdf = create_trip_lines(filtered_df)
    
    if trip_lines_gdf.empty:
        return valid_trips, 0, 0, 0
    
    simplified_gdf = simplify_trip_lines(trip_lines_gdf)
    
    # Try clustering
    n_clusters, n_clustered_routes = cluster_routes(
        simplified_gdf, 
        params['cluster_eps'], 
        params['cluster_min_samples']
    )
    
    return valid_trips, len(trip_lines_gdf), n_clusters, n_clustered_routes

def binary_search_tune(df):
    """
    Use binary search to find optimal parameters.
    
    Returns:
        dict: Optimal parameters
    """
    print("Starting binary parameter search...")
    
    # Define parameter ranges
    min_speed_range = [0.1, 5.0]  # 0.1 to 5.0 knots
    min_trip_points_range = [2, 20]  # 2 to 20 points
    cluster_eps_range = [0.003, 0.1]  # 0.003 to 0.1 (fine to coarse clustering)
    cluster_min_samples_range = [2, 5]  # 2 to 5 samples
    
    # Initial "ambitious" parameters - aim for detailed clustering
    best_params = {
        'min_speed': 2.0,
        'min_trip_points': 10,
        'cluster_eps': 0.01,
        'cluster_min_samples': 2,
        'by_vessel_type': False,
        'routes_only': True  # Default to routes only
    }
    
    # First, ensure we have valid trips
    print("\nStep 1: Optimizing filtering parameters to ensure we have valid trips...")
    
    # Test if we have valid trips with current parameters
    valid_trips, has_lines, _, _ = evaluate_parameters(df, best_params)
    
    # If we don't have valid trips or lines, relax filtering parameters
    if valid_trips == 0 or has_lines == 0:
        # Try less aggressive filtering
        best_params['min_speed'] = 0.5
        best_params['min_trip_points'] = 3
        
        valid_trips, has_lines, _, _ = evaluate_parameters(df, best_params)
        
        # If still no valid trips, use minimum possible values
        if valid_trips == 0 or has_lines == 0:
            best_params['min_speed'] = min_speed_range[0]
            best_params['min_trip_points'] = min_trip_points_range[0]
            
            valid_trips, has_lines, _, _ = evaluate_parameters(df, best_params)
            
            if valid_trips == 0 or has_lines == 0:
                print("Warning: Unable to find parameters that produce valid trip lines.")
                # Return conservative parameters anyway
                return best_params
    
    print(f"Found parameters that produce {valid_trips} valid trips and {has_lines} line geometries.")
    
    # Now optimize clustering parameters
    print("\nStep 2: Optimizing clustering parameters...")
    
    # Try different cluster_eps values (more important than min_samples)
    eps_values = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    samples_values = [2, 3]
    
    best_clusters = 0
    best_clustered_routes = 0
    
    # Grid search for clustering parameters
    for eps in eps_values:
        for samples in samples_values:
            test_params = best_params.copy()
            test_params['cluster_eps'] = eps
            test_params['cluster_min_samples'] = samples
            
            _, _, n_clusters, n_clustered_routes = evaluate_parameters(df, test_params)
            
            # If we've found clusters, consider this a success
            if n_clusters > best_clusters or (n_clusters == best_clusters and n_clustered_routes > best_clustered_routes):
                best_clusters = n_clusters
                best_clustered_routes = n_clustered_routes
                best_params['cluster_eps'] = eps
                best_params['cluster_min_samples'] = samples
                
                print(f"Found better parameters: eps={eps}, min_samples={samples}")
                print(f"  Clusters: {n_clusters}, Clustered routes: {n_clustered_routes}")
                
                # If we have a good number of clusters, we can stop early
                if n_clusters >= 5 and n_clustered_routes > 10:
                    break
    
    # Finalize parameters
    print("\nFinal optimized parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params

def save_parameters(params, output_file):
    """Save parameters to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # Convert datetime objects to strings
        serializable_params = {k: (str(v) if isinstance(v, datetime) else v) for k, v in params.items()}
        
        with open(output_file, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        
        print(f"Parameters saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving parameters: {e}")
        return False

def main():
    """Main function for parameter tuning."""
    parser = argparse.ArgumentParser(description='Tune DBSCAN parameters for maritime route visualization.')
    parser.add_argument('--input', type=str, help='Input CSV file with vessel positions')
    parser.add_argument('--output', type=str, help='Output JSON file for parameters')
    
    args = parser.parse_args()
    
    # If no input specified, find most recent CSV file
    if not args.input:
        print("No input file specified. Looking for the most recent CSV file...")
        input_file = find_latest_csv()
        if not input_file:
            print("Error: No input file specified and no CSV files found.")
            return
        print(f"Using {input_file} as input file.")
    else:
        input_file = args.input
    
    # If no output specified, use the input filename with .json extension
    if not args.output:
        output_file = os.path.splitext(input_file)[0] + '.json'
    else:
        output_file = args.output
    
    # Load data
    df = load_vessel_data(input_file)
    
    if df.empty:
        print("No data to process. Exiting.")
        return
    
    # Run parameter tuning
    optimal_params = binary_search_tune(df)
    
    # Add metadata to parameters
    optimal_params['input_file'] = input_file
    optimal_params['tuned_on'] = datetime.now()
    
    # Save parameters
    save_parameters(optimal_params, output_file)
    
    print("\nTuning complete. Parameters saved to:", output_file)
    print("Run render.py with these parameters to generate the visualization:")
    print(f"python src/render.py --params {output_file}")

if __name__ == "__main__":
    main()
