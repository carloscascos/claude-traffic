#!/usr/bin/env python3
"""
Cruise Vessel Highway Map Generator

This script generates a maritime highway map showing routes used by cruise vessels
with line widths proportional to gross tonnage. It uses the processed vessel position 
data from the route_data_prep.py script.

Usage:
    python cruise_highway_map.py --input "../data/cruise_routes_oct2024.csv" --output "../data/cruise_highway_map_oct2024.html"
"""

import os
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
# Fix: importing branca.colormap instead of folium.cm
import branca.colormap as cm
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
# Using matplotlib's cm with a different name to avoid conflict
import matplotlib.cm as mpl_cm
from datetime import datetime
import sys

def load_vessel_data(input_file):
    """
    Load vessel position data from CSV file.
    
    Args:
        input_file (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with vessel positions
    """
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
    
    # Filter out positions where the vessel is not moving
    if 'calculated_speed' in df.columns:
        moving_df = df[df['calculated_speed'] >= min_speed].copy()
    else:
        print("Warning: No speed column found. Cannot filter by speed.")
        moving_df = df.copy()
    
    print(f"After speed filtering: {len(moving_df)} records")
    
    # Count points per trip
    trip_counts = moving_df.groupby('unique_trip_id').size()
    
    # Filter out trips with too few points
    valid_trips = trip_counts[trip_counts >= min_trip_points].index
    valid_df = moving_df[moving_df['unique_trip_id'].isin(valid_trips)].copy()
    
    print(f"After trip length filtering: {len(valid_df)} records")
    print(f"Valid trips: {len(valid_trips)}")
    
    return valid_df

def create_trip_lines(df):
    """
    Convert trip points to LineString objects.
    
    Args:
        df (pandas.DataFrame): DataFrame with vessel positions
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with trip lines
    """
    print("Creating trip lines...")
    
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
            imo = trip_points['imo'].iloc[0]
            vessel_name = trip_points['vessel_name'].iloc[0] if 'vessel_name' in trip_points else 'Unknown'
            gross_tonnage = trip_points['GT'].iloc[0] if 'GT' in trip_points else 0
            start_time = trip_points['timestamp'].min()
            end_time = trip_points['timestamp'].max()
            
            trip_lines.append(line)
            trip_data.append({
                'trip_id': trip_id,
                'imo': imo,
                'vessel_name': vessel_name,
                'gross_tonnage': gross_tonnage,
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': (end_time - start_time).total_seconds() / 3600,
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
    """
    Simplify trip lines to reduce complexity.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame with trip lines
        tolerance (float): Simplification tolerance
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with simplified trip lines
    """
    print(f"Simplifying trip lines (tolerance={tolerance})...")
    
    # Create a copy to avoid modifying the original
    simplified_gdf = gdf.copy()
    
    # Simplify geometries
    simplified_gdf['geometry'] = simplified_gdf['geometry'].simplify(tolerance)
    
    return simplified_gdf

def cluster_routes(gdf, eps=0.02, min_samples=2):
    """
    Cluster similar routes using DBSCAN.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame with trip lines
        eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples (int): Minimum number of samples in a neighborhood for a point to be considered a core point
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with route clusters
    """
    print(f"Clustering routes (eps={eps}, min_samples={min_samples})...")
    
    # Function to extract features from LineString
    def extract_line_features(line, num_points=40):  # Increased from 20 to 40
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
    
    # Perform DBSCAN clustering with more lenient parameters
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
    Width will be based on total gross tonnage.
    
    Args:
        clustered_gdf (geopandas.GeoDataFrame): GeoDataFrame with clustered routes
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with representative routes
    """
    print("Aggregating routes by cluster...")
    
    # Filter out unclustered routes
    if 'route_cluster' in clustered_gdf.columns:
        clustered_routes = clustered_gdf[clustered_gdf['route_cluster'] >= 0].copy()
    else:
        print("No route_cluster column found. Cannot aggregate routes.")
        return clustered_gdf
    
    if clustered_routes.empty:
        print("No clustered routes to aggregate.")
        return clustered_gdf
    
    # Aggregate by cluster
    aggregated_routes = []
    
    for cluster_id, cluster_group in clustered_routes.groupby('route_cluster'):
        # Calculate total gross tonnage for this route
        total_gt = cluster_group['gross_tonnage'].sum()
        
        # Calculate average gross tonnage per vessel
        avg_gt = cluster_group['gross_tonnage'].mean()
        
        # Count vessels on this route
        vessel_count = len(cluster_group['imo'].unique())
        
        # Count trips on this route
        trip_count = len(cluster_group)
        
        # Get unique vessel names
        vessel_names = cluster_group['vessel_name'].unique()
        vessel_names_str = ', '.join(vessel_names[:5])
        if len(vessel_names) > 5:
            vessel_names_str += f" and {len(vessel_names) - 5} more"
        
        # Select a representative route (the one with the median length)
        cluster_group['length'] = cluster_group.geometry.length
        median_idx = cluster_group['length'].argsort()[len(cluster_group) // 2]
        representative_line = cluster_group.iloc[median_idx].geometry
        
        # Calculate the average duration
        avg_duration = cluster_group['duration_hours'].mean()
        
        aggregated_routes.append({
            'route_id': cluster_id,
            'vessel_count': vessel_count,
            'trip_count': trip_count,
            'total_gross_tonnage': total_gt,
            'avg_gross_tonnage': avg_gt,
            'geometry': representative_line,
            'vessel_names': vessel_names_str,
            'avg_duration_hours': avg_duration
        })
    
    # Create GeoDataFrame with aggregated routes
    aggregated_gdf = gpd.GeoDataFrame(aggregated_routes, crs=clustered_routes.crs)
    print(f"Created {len(aggregated_gdf)} aggregated routes.")
    
    return aggregated_gdf

def generate_map(routes_gdf, port_gdf=None, output_file='cruise_highway_map.html'):
    """
    Generate a maritime highway map with routes proportional to gross tonnage.
    
    Args:
        routes_gdf (geopandas.GeoDataFrame): GeoDataFrame with aggregated routes
        port_gdf (geopandas.GeoDataFrame, optional): GeoDataFrame with port locations. Defaults to None.
        output_file (str, optional): Output HTML file path. Defaults to 'cruise_highway_map.html'.
    """
    print(f"Generating maritime highway map to {output_file}...")
    
    if routes_gdf.empty:
        print("No routes to display on the map.")
        return
    
    # Check if the GeoDataFrame has aggregated routes
    if 'total_gross_tonnage' not in routes_gdf.columns:
        print("No aggregated routes found. Using individual trips instead.")
        
        # Use individual trips for display
        if 'gross_tonnage' not in routes_gdf.columns:
            print("No gross tonnage information available. Using default line width.")
            routes_gdf['total_gross_tonnage'] = 1000  # Default value
            routes_gdf['avg_gross_tonnage'] = 1000
            routes_gdf['trip_count'] = 1
            routes_gdf['vessel_count'] = 1
            routes_gdf['vessel_names'] = routes_gdf['vessel_name']
            routes_gdf['avg_duration_hours'] = 0
            routes_gdf['route_id'] = routes_gdf.index
        else:
            # Use the individual trips' gross tonnage
            routes_gdf['total_gross_tonnage'] = routes_gdf['gross_tonnage']
            routes_gdf['avg_gross_tonnage'] = routes_gdf['gross_tonnage']
            routes_gdf['trip_count'] = 1
            routes_gdf['vessel_count'] = 1
            routes_gdf['vessel_names'] = routes_gdf['vessel_name']
            routes_gdf['avg_duration_hours'] = routes_gdf['duration_hours']
            routes_gdf['route_id'] = routes_gdf.index
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Find map center (average of route centroids)
    centroids = routes_gdf.geometry.centroid
    map_center = [centroids.y.mean(), centroids.x.mean()]
    
    # Create base map
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB positron')
    
    # Add scale
    folium.plugins.MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m)
    
    # Calculate line widths based on gross tonnage
    min_gt = routes_gdf['total_gross_tonnage'].min()
    max_gt = routes_gdf['total_gross_tonnage'].max()
    
    def get_line_width(gt):
        """Calculate line width based on gross tonnage."""
        # Scale between 2 and 20 pixels (increased max width)
        if max_gt == min_gt:
            return 5  # Default width if all routes have same GT
        # Use a logarithmic scale for better visualization
        return 2 + (np.log(1 + gt - min_gt) / np.log(1 + max_gt - min_gt)) * 18
    
    # Create a color map based on tonnage - using branca.colormap
    colormap = cm.LinearColormap(
        colors=['#1E88E5', '#1E88E5', '#1E88E5', '#1E88E5'],
        index=[min_gt, min_gt + (max_gt-min_gt)/3, min_gt + 2*(max_gt-min_gt)/3, max_gt],
        vmin=min_gt,
        vmax=max_gt
    )
    
    # Create feature group for routes
    fg_routes = folium.FeatureGroup(name='Cruise Routes')
    
    # Add routes to map
    for idx, route in routes_gdf.iterrows():
        # Calculate line width
        width = get_line_width(route['total_gross_tonnage'])
        
        # Calculate color based on gross tonnage - using a single color for now
        color = '#1E88E5'  # Using a fixed color instead of colormap
        
        # Format tooltip
        tooltip = f"Route {route['route_id']}: {route['trip_count']} trips, {int(route['total_gross_tonnage']):,} GT"
        
        # Format popup content
        popup_html = f"""
        <h4>Route {route['route_id']}</h4>
        <b>Trip Count:</b> {route['trip_count']}<br>
        <b>Unique Vessels:</b> {route['vessel_count']}<br>
        <b>Total Gross Tonnage:</b> {int(route['total_gross_tonnage']):,} GT<br>
        <b>Average Gross Tonnage:</b> {int(route['avg_gross_tonnage']):,} GT<br>
        <b>Average Duration:</b> {route['avg_duration_hours']:.1f} hours<br>
        <b>Vessels:</b> {route['vessel_names']}
        """
        
        # Convert LineString to coordinates for Folium
        line_coords = [(point[1], point[0]) for point in list(route.geometry.coords)]
        
        # Add line to map
        folium.PolyLine(
            line_coords,
            color=color,
            weight=width,
            opacity=0.7,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(fg_routes)
    
    # Add routes layer to map
    fg_routes.add_to(m)
    
    # Add ports if provided
    if port_gdf is not None and not port_gdf.empty:
        fg_ports = folium.FeatureGroup(name='Ports')
        
        # Create marker cluster for ports
        marker_cluster = MarkerCluster().add_to(fg_ports)
        
        for idx, port in port_gdf.iterrows():
            # Extract port coordinates
            if port.geometry.geom_type == 'Point':
                lat, lon = port.geometry.y, port.geometry.x
            else:
                # Use centroid for non-point geometries
                lat, lon = port.geometry.centroid.y, port.geometry.centroid.x
            
            # Create port marker
            folium.Marker(
                location=[lat, lon],
                popup=port['port_name'] if 'port_name' in port else f"Port {idx}",
                icon=folium.Icon(color='red', icon='anchor', prefix='fa')
            ).add_to(marker_cluster)
        
        # Add ports layer to map
        fg_ports.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 230px; height: 140px; 
                border: 2px solid grey; z-index: 9999; font-size: 14px;
                background-color: white; padding: 10px; border-radius: 5px;">
        <div style="text-align: center; margin-bottom: 5px;">
            <strong>Cruise Vessel Traffic</strong>
        </div>
        <div style="margin-bottom: 5px;">
            Line width is proportional to gross tonnage
        </div>
        <div>
            <span style="display: inline-block; width: 50px; height: 2px; background-color: #1E88E5;"></span>
            Lower Tonnage
        </div>
        <div>
            <span style="display: inline-block; width: 50px; height: 6px; background-color: #1E88E5;"></span>
            Medium Tonnage
        </div>
        <div>
            <span style="display: inline-block; width: 50px; height: 12px; background-color: #1E88E5;"></span>
            Higher Tonnage
        </div>
    </div>
    '''
    
    # Add legend to map
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title with the date range from the input file
    # Extract date range from output filename or default to Sept-Nov 2024
    if "sept_oct_nov" in output_file.lower():
        date_range = "Sept-Nov 2024"
    elif "oct" in output_file.lower():
        date_range = "October 2024"
    elif "sept" in output_file.lower():
        date_range = "September 2024"
    elif "nov" in output_file.lower():
        date_range = "November 2024"
    else:
        date_range = "Sept-Nov 2024"
    
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%); 
                z-index: 9999; font-size: 18px; font-weight: bold;
                background-color: white; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);">
        Cruise Vessel Maritime Highway Map - {date_range}
    </div>
    '''
    
    # Add title to map
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    m.save(output_file)
    print(f"Map saved to {output_file}")

def create_port_data(df, port_radius=0.05):
    """
    Extract port locations from vessel positions.
    
    Args:
        df (pandas.DataFrame): DataFrame with vessel positions
        port_radius (float): Radius to consider as the same port location (in degrees)
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with port locations
    """
    print(f"Extracting port locations (radius={port_radius})...")
    
    # Filter for slow/stopped vessels
    if 'calculated_speed' in df.columns:
        stopped_df = df[df['calculated_speed'] < 1.0].copy()
    else:
        print("Warning: No speed column found. Using all positions for port detection.")
        stopped_df = df.copy()
    
    if stopped_df.empty:
        print("No stopped positions found for port detection.")
        return gpd.GeoDataFrame()
    
    # Create points
    points = [Point(row['longitude'], row['latitude']) for _, row in stopped_df.iterrows()]
    
    # Create GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        {'geometry': points},
        crs="EPSG:4326"
    )
    
    # Cluster points to identify ports
    coordinates = np.array([[point.x, point.y] for point in points])
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=port_radius, min_samples=5).fit(coordinates)
    
    # Add cluster labels to GeoDataFrame
    points_gdf['cluster'] = clustering.labels_
    
    # Filter out noise points
    clustered_points = points_gdf[points_gdf['cluster'] >= 0]
    
    if clustered_points.empty:
        print("No port clusters found.")
        return gpd.GeoDataFrame()
    
    # Group by cluster and calculate centroid
    port_data = []
    
    for cluster_id, cluster_group in clustered_points.groupby('cluster'):
        # Calculate centroid
        centroid = cluster_group.unary_union.centroid
        
        # Count points in cluster
        point_count = len(cluster_group)
        
        port_data.append({
            'port_id': cluster_id,
            'point_count': point_count,
            'geometry': centroid
        })
    
    # Create GeoDataFrame with port locations
    port_gdf = gpd.GeoDataFrame(port_data, crs="EPSG:4326")
    print(f"Identified {len(port_gdf)} potential port locations.")
    
    return port_gdf

def main():
    """
    Main function to generate a maritime highway map for cruise vessels.
    """
    parser = argparse.ArgumentParser(description='Generate a maritime highway map for cruise vessels.')
    parser.add_argument('--input', type=str, default='../data/cruise_routes_oct2024.csv', help='Input CSV file with vessel positions')
    parser.add_argument('--output', type=str, default='../data/cruise_highway_map_oct2024.html', help='Output HTML file for the map')
    parser.add_argument('--min_speed', type=float, default=3.0, help='Minimum speed (knots) for route points')
    parser.add_argument('--min_trip_points', type=int, default=10, help='Minimum points for a valid trip')
    parser.add_argument('--cluster_eps', type=float, default=0.01, help='DBSCAN eps parameter for route clustering')
    parser.add_argument('--cluster_min_samples', type=int, default=2, help='DBSCAN min_samples parameter for route clustering')
    
    args = parser.parse_args()
    
    # Print the parameters
    print(f"Generating maritime highway map with parameters:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Min speed: {args.min_speed}")
    print(f"  Min trip points: {args.min_trip_points}")
    print(f"  Cluster eps: {args.cluster_eps}")
    print(f"  Cluster min samples: {args.cluster_min_samples}")
    
    # Load the data
    df = load_vessel_data(args.input)
    
    if df.empty:
        print("No data to process. Exiting.")
        return
    
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
    
    # Simplify trip lines
    simplified_gdf = simplify_trip_lines(trip_lines_gdf)
    
    # Cluster routes
    clustered_gdf = cluster_routes(simplified_gdf, args.cluster_eps, args.cluster_min_samples)
    
    # Aggregate routes
    aggregated_gdf = aggregate_routes(clustered_gdf)
    
    # Extract port locations
    port_gdf = create_port_data(df)
    
    # Generate the map
    generate_map(aggregated_gdf, port_gdf, args.output)
    
    print("Maritime highway map generation completed.")

if __name__ == "__main__":
    main()
