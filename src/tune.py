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

def extract_line_features(line, num_points=60):
    """
    Extract features from a LineString by sampling points along the line.
    Uses more points and includes directional information for better similarity detection.
    """
    if not line.is_empty:
        try:
            # Sample more points along the line for better representation
            distances = np.linspace(0, line.length, num_points)
            points = [line.interpolate(distance) for distance in distances]
            
            # Extract coordinates
            features = []
            
            # Add normalized start and end points (higher weight)
            start_x, start_y = line.coords[0]
            end_x, end_y = line.coords[-1]
            features.extend([start_x, start_y, end_x, end_y])
            
            # Add general direction vector
            direction_x = end_x - start_x
            direction_y = end_y - start_y
            # Normalize direction to avoid influencing clustering with route length
            magnitude = np.sqrt(direction_x**2 + direction_y**2)
            if magnitude > 0:
                direction_x /= magnitude
                direction_y /= magnitude
            features.extend([direction_x, direction_y])
            
            # Add all sampled points
            for point in points:
                features.extend([point.x, point.y])
            
            # Add line bounding box (min/max coordinates)
            minx, miny, maxx, maxy = line.bounds
            features.extend([minx, miny, maxx, maxy])
            
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    return None

def cluster_routes(gdf, eps, min_samples, by_vessel_type=False):
    """
    Cluster similar routes using DBSCAN with enhanced feature extraction and metrics.
    """
    print(f"Clustering routes (eps={eps:.4f}, min_samples={min_samples})...")
    
    if by_vessel_type and 'vessel_type' in gdf.columns:
        # Cluster separately by vessel type (not our focus for tuning)
        raise NotImplementedError("By-vessel-type clustering not implemented for parameter tuning")
    
    # Extract features from each line
    features = []
    valid_indices = []
    
    print(f"Extracting features from {len(gdf)} route lines...")
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
    
    # Check feature dimensions
    feature_length = features_array.shape[1]
    print(f"Feature vector length: {feature_length}")
    
    # Normalize features to prevent any dimension from dominating
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    
    # For very small datasets, reduce min_samples to ensure we get some clustering
    if len(features_array) < 10 and min_samples > 2:
        print(f"Small dataset detected ({len(features_array)} routes). Reducing min_samples to 2.")
        min_samples = 2
    
    # Perform DBSCAN clustering with detailed output
    print(f"Running DBSCAN with {len(features_array)} routes...")
    clustering = DBSCAN(
        eps=eps, 
        min_samples=min_samples, 
        metric='euclidean',
        n_jobs=-1  # Use all available cores
    ).fit(features_array)
    
    # Get cluster labels
    labels = clustering.labels_
    
    # Count clusters and analyze distribution
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_unclustered = (labels == -1).sum()
    n_clustered = len(labels) - n_unclustered
    
    # Print detailed clustering statistics
    print(f"Found {n_clusters} route clusters. {n_unclustered} routes were not clustered.")
    
    if n_clusters > 0:
        # Count routes per cluster
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_sizes[label] = (labels == label).sum()
        
        # Print cluster sizes
        sizes_str = ", ".join([f"Cluster {k}: {v} routes" for k, v in cluster_sizes.items()])
        print(f"Cluster sizes: {sizes_str}")
        
        # Calculate clustering efficiency
        clustering_ratio = n_clustered / len(labels)
        print(f"Clustering efficiency: {clustering_ratio:.2f} ({n_clustered}/{len(labels)} routes)")
    
    return n_clusters, n_clustered

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
    
    # Add more detailed logging
    total_routes = len(trip_lines_gdf)
    unclustered_routes = total_routes - n_clustered_routes
    clustering_percentage = (n_clustered_routes / total_routes * 100) if total_routes > 0 else 0
    
    print(f"Clustering efficiency: {n_clustered_routes}/{total_routes} routes clustered ({clustering_percentage:.1f}%)")
    
    if n_clusters > 0:
        avg_routes_per_cluster = n_clustered_routes / n_clusters
        print(f"Average routes per cluster: {avg_routes_per_cluster:.1f}")
    
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
        'by_vessel_type': True,  # Always enable vessel type layers
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
    
    # Use a wider range of eps values with exponential increase
    # This allows testing both fine-grained and coarse clustering
    eps_values = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3]
    samples_values = [2, 3, 4, 5]
    
    best_clusters = 0
    best_clustered_routes = 0
    best_cluster_ratio = 0  # Ratio of clustered routes to total routes
    
    # First pass: try to find parameters that produce ANY clusters
    print("First pass: Testing various eps values to find initial clustering...")
    for eps in eps_values:
        # Start with min_samples=2 for initial testing
        test_params = best_params.copy()
        test_params['cluster_eps'] = eps
        test_params['cluster_min_samples'] = 2
        
        valid_trips, has_lines, n_clusters, n_clustered_routes = evaluate_parameters(df, test_params)
        
        if n_clusters > 0:
            print(f"Found initial clustering with eps={eps}, min_samples=2")
            print(f"  Clusters: {n_clusters}, Clustered routes: {n_clustered_routes}")
            
            # Save these as our starting point
            best_clusters = n_clusters
            best_clustered_routes = n_clustered_routes
            best_params['cluster_eps'] = eps
            best_params['cluster_min_samples'] = 2
            
            # Calculate ratio of clustered routes to total valid routes
            if has_lines > 0:
                best_cluster_ratio = n_clustered_routes / has_lines
            
            # We found a working value, break from first pass
            break
    
    # If we didn't find any clustering, try with larger eps values
    if best_clusters == 0:
        print("No clustering found in first pass. Trying with larger eps values...")
        larger_eps_values = [0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
        
        for eps in larger_eps_values:
            test_params = best_params.copy()
            test_params['cluster_eps'] = eps
            test_params['cluster_min_samples'] = 2
            
            valid_trips, has_lines, n_clusters, n_clustered_routes = evaluate_parameters(df, test_params)
            
            if n_clusters > 0:
                print(f"Found initial clustering with eps={eps}, min_samples=2")
                print(f"  Clusters: {n_clusters}, Clustered routes: {n_clustered_routes}")
                
                # Save these as our starting point
                best_clusters = n_clusters
                best_clustered_routes = n_clustered_routes
                best_params['cluster_eps'] = eps
                best_params['cluster_min_samples'] = 2
                
                # Calculate ratio
                if has_lines > 0:
                    best_cluster_ratio = n_clustered_routes / has_lines
                
                break
    
    # If we found initial clustering, refine around that value
    if best_clusters > 0:
        print("\nSecond pass: Refining parameters around initial successful value...")
        
        # Get the successful eps value
        successful_eps = best_params['cluster_eps']
        
        # Create a range of values around the successful one (±50%)
        min_eps = max(0.001, successful_eps * 0.5)
        max_eps = successful_eps * 1.5
        refined_eps_values = np.linspace(min_eps, max_eps, 8)
        
        # Grid search with refined values
        for eps in refined_eps_values:
            for samples in samples_values:
                test_params = best_params.copy()
                test_params['cluster_eps'] = eps
                test_params['cluster_min_samples'] = samples
                
                valid_trips, has_lines, n_clusters, n_clustered_routes = evaluate_parameters(df, test_params)
                
                # Calculate cluster ratio
                cluster_ratio = n_clustered_routes / has_lines if has_lines > 0 else 0
                
                # Better parameters found if:
                # 1. More clusters OR
                # 2. Same clusters but more routes clustered OR
                # 3. Better cluster ratio (balance between too many small clusters and too few large ones)
                if (n_clusters > best_clusters) or \
                   (n_clusters == best_clusters and n_clustered_routes > best_clustered_routes) or \
                   (n_clusters >= 2 and cluster_ratio > best_cluster_ratio and cluster_ratio <= 0.9):
                    
                    best_clusters = n_clusters
                    best_clustered_routes = n_clustered_routes
                    best_cluster_ratio = cluster_ratio
                    best_params['cluster_eps'] = eps
                    best_params['cluster_min_samples'] = samples
                    
                    print(f"Found better parameters: eps={eps:.4f}, min_samples={samples}")
                    print(f"  Clusters: {n_clusters}, Clustered routes: {n_clustered_routes}")
                    print(f"  Cluster ratio: {cluster_ratio:.2f}")
                    
                    # If we have a good number of clusters and good ratio, we can stop early
                    if n_clusters >= 3 and 0.3 <= cluster_ratio <= 0.8:
                        print("Found optimal parameters with good balance. Stopping search.")
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
