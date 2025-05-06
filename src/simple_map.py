#!/usr/bin/env python3
"""
Simple Maritime Highway Map Generator

This script creates a basic map of vessel routes without complex DBSCAN clustering.
"""

import pandas as pd
import folium
from folium import plugins  # Import plugins explicitly
import numpy as np
from shapely.geometry import LineString
import geopandas as gpd
import os

def load_and_process_data(input_file, min_speed=0.2, min_points=5):
    """Load vessel data and do basic processing"""
    print(f"Loading data from {input_file}...")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        return None
    
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records.")
        
        # Convert timestamp to datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by speed
        if 'calculated_speed' in df.columns:
            df = df[df['calculated_speed'] >= min_speed].copy()
            print(f"After speed filtering: {len(df)} records")
        
        # Group by trip and filter those with too few points
        trip_sizes = df.groupby('unique_trip_id').size()
        valid_trips = trip_sizes[trip_sizes >= min_points].index
        df = df[df['unique_trip_id'].isin(valid_trips)].copy()
        print(f"After min points filtering: {len(df)} records")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_routes(df):
    """Create GeoDataFrame with route lines"""
    print("Creating routes...")
    
    # Create lines for each trip
    routes = []
    
    for trip_id, group in df.groupby('unique_trip_id'):
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Need at least 2 points to create a line
        if len(group) < 2:
            continue
            
        # Create line coordinates
        line_coords = [(row['longitude'], row['latitude']) for _, row in group.iterrows()]
        line = LineString(line_coords)
        
        # Get trip data
        vessel_name = group['vessel_name'].iloc[0]
        gt = group['GT'].iloc[0]
        start_time = group['timestamp'].min()
        end_time = group['timestamp'].max()
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        
        # Add to routes list
        routes.append({
            'trip_id': trip_id,
            'vessel_name': vessel_name,
            'gross_tonnage': gt,
            'duration_hours': duration,
            'imo': group['imo'].iloc[0],
            'geometry': line
        })
    
    # Create GeoDataFrame
    if not routes:
        print("No valid routes created.")
        return None
        
    routes_gdf = gpd.GeoDataFrame(routes, crs="EPSG:4326")
    print(f"Created {len(routes_gdf)} routes.")
    
    return routes_gdf

def create_map(routes_gdf, output_file):
    """Create the map visualization"""
    print(f"Creating map and saving to {output_file}...")
    
    # Get center of map
    # Convert to a projected CRS first to get accurate centroid
    projected_gdf = routes_gdf.to_crs('EPSG:3857')  # Web Mercator
    centroids = projected_gdf.geometry.centroid
    # Convert back to WGS84 for folium
    centroids_wgs84 = gpd.GeoSeries(centroids, crs='EPSG:3857').to_crs('EPSG:4326')
    map_center = [centroids_wgs84.y.mean(), centroids_wgs84.x.mean()]
    
    # Create map
    m = folium.Map(location=map_center, zoom_start=4, tiles='CartoDB positron')
    
    # Skip the measure control as it's causing issues
    # plugins.MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m)
    
    # Calculate line width based on gross tonnage
    min_gt = routes_gdf['gross_tonnage'].min()
    max_gt = routes_gdf['gross_tonnage'].max()
    
    def get_line_width(gt):
        """Calculate width based on tonnage"""
        if max_gt == min_gt:
            return 3
        return 1 + (np.log(1 + gt - min_gt) / np.log(1 + max_gt - min_gt)) * 10
    
    # Add routes to map
    for _, route in routes_gdf.iterrows():
        # Calculate width
        width = get_line_width(route['gross_tonnage'])
        
        # Create tooltip
        tooltip = f"{route['vessel_name']} - {int(route['gross_tonnage']):,} GT"
        
        # Create popup
        popup_html = f"""
        <h4>{route['vessel_name']}</h4>
        <b>IMO:</b> {route['imo']}<br>
        <b>Gross Tonnage:</b> {int(route['gross_tonnage']):,} GT<br>
        <b>Duration:</b> {route['duration_hours']:.1f} hours
        """
        
        # Convert coordinates for folium
        coords = [(point[1], point[0]) for point in list(route.geometry.coords)]
        
        # Add line to map
        folium.PolyLine(
            coords,
            color='#1E88E5',
            weight=width,
            opacity=0.7,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%); 
                z-index: 9999; font-size: 18px; font-weight: bold;
                background-color: white; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);">
        Cruise Vessel Maritime Highway Map - Sept-Nov 2024
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
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
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_file)
    print(f"Map saved to {output_file}")

def main():
    """Main function"""
    input_file = "data/cruise_routes_sample.csv"  # Using the sample data
    output_file = "data/maritime_routes_simple.html"
    
    # Load and process data
    df = load_and_process_data(input_file)
    if df is None:
        return
    
    # Create routes
    routes_gdf = create_routes(df)
    if routes_gdf is None:
        return
    
    # Create map
    create_map(routes_gdf, output_file)
    print("Done!")

if __name__ == "__main__":
    main()
