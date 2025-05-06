#!/usr/bin/env python3
"""
Visualization Rendering Module

This module creates geospatial visualizations of maritime routes,
generates charts and graphs of vessel statistics, and produces
map overlays with traffic patterns.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import folium
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm
from shapely.geometry import LineString, Point
import geopandas as gpd
from sklearn.cluster import DBSCAN
from datetime import datetime

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

def load_parameters(params_file):
    """
    Load tuned parameters from JSON file.
    
    Args:
        params_file (str): Path to the JSON file
        
    Returns:
        dict: Dictionary with parameters
    """
    print(f"Loading parameters from {params_file}...")
    
    # Default parameters
    default_params = {
        'clustering': {
            'eps': 0.02,
            'min_samples': 3
        },
        'visualization': {
            'zoom_level': 8,
            'line_opacity': 0.7,
            'min_line_width': 2,
            'max_line_width': 10,
            'use_log_scale': True,
            'cluster_routes': True,
            'fleet_filtering': True,
            'fleet_colors': {
                'Cruise/Passenger': '#E41A1C',
                'Container': '#377EB8',
                'Bulk Carrier': '#4DAF4A',
                'Tanker': '#984EA3',
                'Ferry/Ro-Ro': '#FF7F00',
                'Fishing': '#FFFF33',
                'Pleasure': '#A65628',
                'Tug': '#F781BF',
                'Other': '#999999',
                'Unknown': '#000000'
            }
        }
    }
    
    if not os.path.exists(params_file):
        print(f"Parameters file {params_file} not found. Using default parameters.")
        return default_params
    
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"Parameters loaded successfully.")
        return params
    except Exception as e:
        print(f"Error loading parameters: {e}")
        print(f"Using default parameters instead.")
        return default_params

def filter_data(df, min_speed=1.0, min_trip_points=10):
    """
    Filter the data to remove stationary positions and short trips.
    
    Args:
        df (pandas.DataFrame): DataFrame with vessel positions
        min_speed (float): Minimum speed (in knots) for a position to be considered moving
        min_trip_points (int): Minimum number of points for a trip to be considered valid
        
    Returns:
        pandas.DataFrame: Filtered DataFrame
    """
    print(f"Filtering data (min_speed={min_speed}, min_trip_points={min_trip_points})...")
    
    if df.empty:
        return df
    
    # Filter out positions where the vessel is not moving
    if 'calculated_speed' in df.columns:
        moving_df = df[df['calculated_speed'] >= min_speed].copy()
        print(f"After speed filtering: {len(moving_df)} records")
    else:
        print("Warning: No speed column found. Cannot filter by speed.")
        moving_df = df.copy()
    
    # Count points per trip
    if 'unique_trip_id' in moving_df.columns:
        trip_counts = moving_df.groupby('unique_trip_id').size()
        
        # Filter out trips with too few points
        valid_trips = trip_counts[trip_counts >= min_trip_points].index
        valid_df = moving_df[moving_df['unique_trip_id'].isin(valid_trips)].copy()
        
        print(f"After trip length filtering: {len(valid_df)} records")
        print(f"Valid trips: {len(valid_trips)}")
        
        return valid_df
    else:
        print("Warning: No unique_trip_id column found. Cannot filter by trip length.")
        return moving_df

def create_trip_lines(df):
    """
    Convert trip points to LineString objects.
    
    Args:
        df (pandas.DataFrame): DataFrame with vessel positions
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with trip lines
    """
    print("Creating trip lines...")
    
    if df.empty:
        return gpd.GeoDataFrame()
    
    # Check if we have required columns
    required_columns = ['unique_trip_id', 'longitude', 'latitude', 'timestamp']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return gpd.GeoDataFrame()
    
    # Group by trip ID and create LineString for each trip
    trip_lines = []
    trip_data = []
    
    for trip_id, trip_points in df.groupby('unique_trip_id'):
        # Sort points by timestamp
        trip_points = trip_points.sort_values('timestamp')
        
        # Create line coordinates
        line_coords = [(row['longitude'], row['latitude']) for _, row in trip_points.iterrows()]
        
        if len(line_coords) >= 2:
            # Create LineString
            line = LineString(line_coords)
            
            # Add trip data
            trip_info = {
                'trip_id': trip_id,
                'geometry': line,
                'point_count': len(trip_points)
            }
            
            # Add additional information if available
            for col in ['imo', 'vessel_name', 'fleet', 'GT']:
                if col in trip_points.columns:
                    trip_info[col] = trip_points[col].iloc[0]
            
            # Calculate timing information
            trip_info['start_time'] = trip_points['timestamp'].min()
            trip_info['end_time'] = trip_points['timestamp'].max()
            trip_info['duration_hours'] = (trip_info['end_time'] - trip_info['start_time']).total_seconds() / 3600
            
            trip_lines.append(line)
            trip_data.append(trip_info)
    
    # Create GeoDataFrame with trip lines
    if not trip_lines:
        print("No valid trip lines created.")
        return gpd.GeoDataFrame()
    
    gdf = gpd.GeoDataFrame(trip_data, geometry=trip_lines, crs="EPSG:4326")
    print(f"Created {len(gdf)} trip lines.")
    
    return gdf

def cluster_routes(gdf, eps=0.02, min_samples=2):
    """
    Cluster similar routes using DBSCAN.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame with trip lines
        eps (float): DBSCAN eps parameter
        min_samples (int): DBSCAN min_samples parameter
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with route clusters
    """
    print(f"Clustering routes (eps={eps}, min_samples={min_samples})...")
    
    if gdf.empty:
        return gdf
    
    # Function to extract features from LineString
    def extract_line_features(line, num_points=20):
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
        return gdf
    
    # Convert to numpy array
    features_array = np.array(features)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features_array)
    
    # Extract valid rows and add cluster labels
    valid_gdf = gdf.loc[valid_indices].copy()
    valid_gdf['route_cluster'] = clustering.labels_
    
    # Count clusters
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    print(f"Found {n_clusters} route clusters. {(clustering.labels_ == -1).sum()} routes were not clustered.")
    
    return valid_gdf

def aggregate_routes(clustered_gdf):
    """
    Aggregate routes by cluster to create representative lines.
    
    Args:
        clustered_gdf (geopandas.GeoDataFrame): GeoDataFrame with clustered routes
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with representative routes
    """
    print("Aggregating routes by cluster...")
    
    if clustered_gdf.empty:
        return clustered_gdf
    
    # Filter out unclustered routes
    if 'route_cluster' in clustered_gdf.columns:
        clustered_routes = clustered_gdf[clustered_gdf['route_cluster'] >= 0].copy()
        unclustered_routes = clustered_gdf[clustered_gdf['route_cluster'] < 0].copy()
        unclustered_routes['is_aggregated'] = False
    else:
        print("No route_cluster column found. Cannot aggregate routes.")
        clustered_gdf['is_aggregated'] = False
        return clustered_gdf
    
    if clustered_routes.empty:
        print("No clustered routes to aggregate.")
        clustered_gdf['is_aggregated'] = False
        return clustered_gdf
    
    # Aggregate by cluster
    aggregated_routes = []
    
    for cluster_id, cluster_group in clustered_routes.groupby('route_cluster'):
        # Dictionary to hold aggregated route data
        aggregated_route = {
            'route_id': f"cluster_{cluster_id}",
            'route_cluster': cluster_id,
            'trip_count': len(cluster_group),
            'vessel_count': len(cluster_group['imo'].unique()) if 'imo' in cluster_group.columns else 0,
            'is_aggregated': True
        }
        
        # Aggregate statistics
        for col in ['GT', 'duration_hours']:
            if col in cluster_group.columns:
                aggregated_route[f"total_{col}"] = cluster_group[col].sum()
                aggregated_route[f"avg_{col}"] = cluster_group[col].mean()
        
        # Calculate fleet distribution
        if 'fleet' in cluster_group.columns:
            fleet_counts = cluster_group['fleet'].value_counts()
            primary_fleet = fleet_counts.idxmax() if not fleet_counts.empty else 'Unknown'
            aggregated_route['fleet'] = primary_fleet
            aggregated_route['fleet_distribution'] = fleet_counts.to_dict()
        else:
            aggregated_route['fleet'] = 'Unknown'
        
        # Get vessel names
        if 'vessel_name' in cluster_group.columns:
            vessel_names = cluster_group['vessel_name'].unique()
            vessel_names_str = ', '.join(vessel_names[:5])
            if len(vessel_names) > 5:
                vessel_names_str += f" and {len(vessel_names) - 5} more"
            aggregated_route['vessel_names'] = vessel_names_str
        
        # Select a representative route (the one with the median length)
        cluster_group['length'] = cluster_group.geometry.length
        median_idx = cluster_group['length'].argsort()[len(cluster_group) // 2]
        representative_line = cluster_group.iloc[median_idx].geometry
        aggregated_route['geometry'] = representative_line
        
        aggregated_routes.append(aggregated_route)
    
    # Create GeoDataFrame with aggregated routes
    if aggregated_routes:
        aggregated_gdf = gpd.GeoDataFrame(aggregated_routes, crs=clustered_routes.crs)
        print(f"Created {len(aggregated_gdf)} aggregated routes.")
        
        # Combine aggregated routes with unclustered routes
        unclustered_routes['route_id'] = unclustered_routes.index.astype(str)
        result_gdf = pd.concat([aggregated_gdf, unclustered_routes], ignore_index=True)
        return result_gdf
    else:
        print("No aggregated routes created.")
        clustered_gdf['is_aggregated'] = False
        return clustered_gdf

def generate_map(routes_gdf, params, output_file='maritime_routes_map.html'):
    """
    Generate an interactive maritime traffic map.
    
    Args:
        routes_gdf (geopandas.GeoDataFrame): GeoDataFrame with routes
        params (dict): Dictionary with visualization parameters
        output_file (str): Output HTML file path
    """
    print(f"Generating maritime route map to {output_file}...")
    
    if routes_gdf.empty:
        print("No routes to display on the map.")
        return
    
    # Extract visualization settings
    vis_params = params.get('visualization', {})
    zoom_level = vis_params.get('zoom_level', 8)
    line_opacity = vis_params.get('line_opacity', 0.7)
    min_line_width = vis_params.get('min_line_width', 2)
    max_line_width = vis_params.get('max_line_width', 10)
    use_log_scale = vis_params.get('use_log_scale', True)
    fleet_filtering = vis_params.get('fleet_filtering', True)
    fleet_colors = vis_params.get('fleet_colors', {})
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Find map center (average of route centroids)
    centroids = routes_gdf.geometry.centroid
    map_center = [centroids.y.mean(), centroids.x.mean()]
    
    # Create base map
    m = folium.Map(location=map_center, zoom_start=zoom_level, tiles='CartoDB positron')
    
    # Add scale
    folium.plugins.MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m)
    
    # Create feature groups for each fleet type if fleet filtering is enabled
    fleet_groups = {}
    if fleet_filtering and 'fleet' in routes_gdf.columns:
        # Get unique fleet types
        fleet_types = routes_gdf['fleet'].unique()
        
        # Create a feature group for each fleet type
        for fleet in fleet_types:
            if pd.notna(fleet) and fleet:  # Skip empty/NaN values
                fleet_groups[fleet] = folium.FeatureGroup(name=f"{fleet}")
    
    # Create a default feature group for all routes if no fleet filtering
    all_routes = folium.FeatureGroup(name='All Routes')
    
    # Function to calculate line width based on gross tonnage
    def get_line_width(gt):
        """Calculate line width based on gross tonnage."""
        if pd.isna(gt) or gt <= 0:
            return min_line_width
        
        # Get GT values to scale
        if 'total_GT' in routes_gdf.columns:
            min_gt = routes_gdf['total_GT'].min()
            max_gt = routes_gdf['total_GT'].max()
        elif 'GT' in routes_gdf.columns:
            min_gt = routes_gdf['GT'].min()
            max_gt = routes_gdf['GT'].max()
        else:
            return min_line_width
        
        # Handle case where all routes have same GT
        if max_gt == min_gt:
            return (min_line_width + max_line_width) / 2
        
        # Scale logarithmically if specified, otherwise linearly
        if use_log_scale:
            return min_line_width + (np.log(1 + gt - min_gt) / np.log(1 + max_gt - min_gt)) * (max_line_width - min_line_width)
        else:
            return min_line_width + ((gt - min_gt) / (max_gt - min_gt)) * (max_line_width - min_line_width)
    
    # Add routes to map
    for idx, route in routes_gdf.iterrows():
        # Get route gross tonnage (either total or individual)
        if 'total_GT' in route and pd.notna(route['total_GT']):
            gt = route['total_GT']
        elif 'GT' in route and pd.notna(route['GT']):
            gt = route['GT']
        else:
            gt = 1000  # Default value
        
        # Calculate line width
        width = get_line_width(gt)
        
        # Get fleet type and color
        if 'fleet' in route and pd.notna(route['fleet']):
            fleet = route['fleet']
            color = fleet_colors.get(fleet, '#1E88E5')  # Default to blue if color not specified
        else:
            fleet = 'Unknown'
            color = fleet_colors.get('Unknown', '#000000')  # Default to black
        
        # Format tooltip based on whether this is an aggregated route or single trip
        if 'is_aggregated' in route and route['is_aggregated']:
            tooltip = f"Route {route['route_id']}: {route['trip_count']} trips, {route['vessel_count']} vessels"
        else:
            vessel_name = route.get('vessel_name', 'Unknown vessel')
            tooltip = f"{vessel_name}: {int(gt):,} GT"
        
        # Format popup content
        popup_html = "<div style='width: 300px; max-height: 200px; overflow-y: auto;'>"
        
        if 'is_aggregated' in route and route['is_aggregated']:
            # Popup for aggregated route
            popup_html += f"<h4>Route Cluster {route['route_cluster']}</h4>"
            popup_html += f"<b>Trip Count:</b> {route['trip_count']}<br>"
            popup_html += f"<b>Unique Vessels:</b> {route['vessel_count']}<br>"
            
            if 'total_GT' in route:
                popup_html += f"<b>Total Gross Tonnage:</b> {int(route['total_GT']):,} GT<br>"
            
            if 'avg_GT' in route:
                popup_html += f"<b>Average Gross Tonnage:</b> {int(route['avg_GT']):,} GT<br>"
            
            if 'avg_duration_hours' in route:
                popup_html += f"<b>Average Duration:</b> {route['avg_duration_hours']:.1f} hours<br>"
            
            if 'vessel_names' in route:
                popup_html += f"<b>Vessels:</b> {route['vessel_names']}<br>"
            
            if 'fleet_distribution' in route:
                popup_html += "<b>Fleet Distribution:</b><br>"
                popup_html += "<ul style='margin-top: 2px; padding-left: 20px;'>"
                for fleet_type, count in route['fleet_distribution'].items():
                    popup_html += f"<li>{fleet_type}: {count} ({count/route['trip_count']*100:.1f}%)</li>"
                popup_html += "</ul>"
        else:
            # Popup for individual trip
            if 'vessel_name' in route:
                popup_html += f"<h4>{route['vessel_name']}</h4>"
            
            if 'imo' in route:
                popup_html += f"<b>IMO:</b> {route['imo']}<br>"
            
            if 'fleet' in route:
                popup_html += f"<b>Fleet:</b> {route['fleet']}<br>"
            
            if 'GT' in route:
                popup_html += f"<b>Gross Tonnage:</b> {int(route['GT']):,} GT<br>"
            
            if 'duration_hours' in route:
                popup_html += f"<b>Trip Duration:</b> {route['duration_hours']:.1f} hours<br>"
            
            if 'start_time' in route and 'end_time' in route:
                start_time = route['start_time'].strftime('%Y-%m-%d %H:%M')
                end_time = route['end_time'].strftime('%Y-%m-%d %H:%M')
                popup_html += f"<b>Period:</b> {start_time} to {end_time}<br>"
        
        popup_html += "</div>"
        
        # Convert LineString to coordinates for Folium
        line_coords = [(point[1], point[0]) for point in list(route.geometry.coords)]
        
        # Create the line object
        line = folium.PolyLine(
            line_coords,
            color=color,
            weight=width,
            opacity=line_opacity,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=300)
        )
        
        # Add to appropriate feature group
        if fleet_filtering and fleet in fleet_groups:
            line.add_to(fleet_groups[fleet])
        else:
            line.add_to(all_routes)
    
    # Add feature groups to the map
    if fleet_filtering and fleet_groups:
        for fleet, group in fleet_groups.items():
            group.add_to(m)
    else:
        all_routes.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px;
                border: 2px solid grey; z-index: 9999; font-size: 14px;
                background-color: white; padding: 10px; border-radius: 5px;">
        <div style="text-align: center; margin-bottom: 5px;">
            <strong>Maritime Traffic Map</strong>
        </div>
        <div style="margin-bottom: 5px;">
            Line width is proportional to gross tonnage
        </div>
    '''
    
    # Add fleet colors to legend if fleet filtering is enabled
    if fleet_filtering and fleet_groups:
        for fleet, color in fleet_colors.items():
            if fleet in fleet_groups:
                legend_html += f'''
                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                    <span style="display: inline-block; width: 20px; height: 3px; background-color: {color}; margin-right: 5px;"></span>
                    <span>{fleet}</span>
                </div>
                '''
    else:
        # Generic legend for line width
        legend_html += f'''
        <div>
            <span style="display: inline-block; width: 50px; height: 2px; background-color: #1E88E5;"></span>
            Lower Tonnage
        </div>
        <div>
            <span style="display: inline-block; width: 50px; height: 6px; background-color: #1E88E5;"></span>
            Medium Tonnage
        </div>
        <div>
            <span style="display: inline-block; width: 50px; height: 10px; background-color: #1E88E5;"></span>
            Higher Tonnage
        </div>
        '''
    
    legend_html += '</div>'
    
    # Add legend to map
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%); 
                z-index: 9999; font-size: 18px; font-weight: bold;
                background-color: white; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);">
        Maritime Traffic Routes
    </div>
    '''
    
    # Add title to map
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    m.save(output_file)
    print(f"Map saved to {output_file}")
    return m

def main():
    """
    Main function to generate visualizations.
    """
    parser = argparse.ArgumentParser(description='Generate maritime traffic visualizations.')
    parser.add_argument('--data', type=str, required=True, help='Input CSV file with processed data')
    parser.add_argument('--params', type=str, help='JSON file with tuned parameters')
    parser.add_argument('--output', type=str, default='output/maritime_routes_map.html', 
                       help='Output file (HTML for map)')
    parser.add_argument('--min-speed', type=float, default=1.0, 
                       help='Minimum speed (knots) for route points')
    parser.add_argument('--min-trip-points', type=int, default=10, 
                       help='Minimum points for a valid trip')
    
    args = parser.parse_args()
    
    # Print the parameters
    print(f"Generating visualizations with parameters:")
    print(f"  Data: {args.data}")
    print(f"  Parameters: {args.params if args.params else 'Using defaults'}")
    print(f"  Output: {args.output}")
    print(f"  Min speed: {args.min_speed}")
    print(f"  Min trip points: {args.min_trip_points}")
    
    # Load the data
    df = load_data(args.data)
    
    if df.empty:
        print("No data to process. Exiting.")
        return
    
    # Load parameters
    params = load_parameters(args.params) if args.params else {}
    
    # Filter the data
    filtered_df = filter_data(df, args.min_speed, args.min_trip_points)
    
    if filtered_df.empty:
        print("No data left after filtering. Exiting.")
        return
    
    # Create trip lines
    trip_lines_gdf = create_trip_lines(filtered_df)
    
    if trip_lines_gdf.empty:
        print("No trip lines created. Exiting.")
        return
    
    # If clustering is enabled in the parameters, perform route clustering
    cluster_routes_enabled = params.get('visualization', {}).get('cluster_routes', True)
    
    if cluster_routes_enabled:
        # Get clustering parameters
        clustering_params = params.get('clustering', {})
        eps = clustering_params.get('eps', 0.02)
        min_samples = clustering_params.get('min_samples', 3)
        
        # Cluster routes
        clustered_gdf = cluster_routes(trip_lines_gdf, eps, min_samples)
        
        # Aggregate routes
        routes_gdf = aggregate_routes(clustered_gdf)
    else:
        # Skip clustering and use individual trip lines
        routes_gdf = trip_lines_gdf
        routes_gdf['is_aggregated'] = False
    
    # Generate the map
    generate_map(routes_gdf, params, args.output)
    
    print("Visualization generation completed.")

if __name__ == "__main__":
    main()
