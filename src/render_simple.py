#!/usr/bin/env python3
"""
Simple Maritime Route Renderer

This script creates a basic map of vessel routes with fleet-based filtering.
It's a simplified version that bypasses the complex clustering process.
"""

import os
import argparse
import pandas as pd
import folium
from folium import plugins  # Fixed import
import branca.colormap as cm
from shapely.geometry import LineString
import geopandas as gpd
import json

def load_data(input_file):
    """Load data from CSV file"""
    print(f"Loading data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records")
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def filter_data(df, min_speed=0.1, min_trip_points=3):
    """Filter data based on speed and trip length"""
    print(f"Filtering data (min_speed={min_speed}, min_trip_points={min_trip_points})...")
    
    if df.empty:
        return df
    
    # Filter by speed if the column exists
    if 'calculated_speed' in df.columns:
        moving_df = df[df['calculated_speed'] >= min_speed].copy()
        print(f"After speed filtering: {len(moving_df)} records")
    else:
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
        print("Warning: No unique_trip_id column found")
        return moving_df

def create_trip_lines(df):
    """Convert points to LineString geometries"""
    print("Creating trip lines...")
    
    if df.empty:
        return gpd.GeoDataFrame()
    
    # Check for required columns
    required_cols = ['unique_trip_id', 'longitude', 'latitude', 'timestamp']
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            return gpd.GeoDataFrame()
    
    # Create lines for each trip
    trip_lines = []
    trip_data = []
    
    for trip_id, group in df.groupby('unique_trip_id'):
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Extract coordinates
        coords = [(row['longitude'], row['latitude']) for _, row in group.iterrows()]
        
        if len(coords) >= 2:
            # Create LineString
            line = LineString(coords)
            
            # Prepare trip data
            trip_info = {
                'trip_id': trip_id,
                'geometry': line,
                'point_count': len(group)
            }
            
            # Add available metadata
            for col in ['imo', 'vessel_name', 'fleet', 'GT']:
                if col in group.columns:
                    trip_info[col] = group[col].iloc[0]
            
            # Add timing info
            trip_info['start_time'] = group['timestamp'].min()
            trip_info['end_time'] = group['timestamp'].max()
            trip_info['duration_hours'] = (trip_info['end_time'] - trip_info['start_time']).total_seconds() / 3600
            
            trip_lines.append(line)
            trip_data.append(trip_info)
    
    # Create GeoDataFrame
    if not trip_lines:
        print("No valid trip lines created")
        return gpd.GeoDataFrame()
        
    gdf = gpd.GeoDataFrame(trip_data, geometry=trip_lines, crs="EPSG:4326")
    print(f"Created {len(gdf)} trip lines")
    
    return gdf

def fleet_colors():
    """Define colors for different fleet types"""
    return {
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

def create_map(routes_gdf, output_file):
    """Create the interactive map"""
    print(f"Generating map to {output_file}...")
    
    if routes_gdf.empty:
        print("No routes to display")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Find map center
    centroids = routes_gdf.geometry.centroid
    map_center = [centroids.y.mean(), centroids.x.mean()]
    
    # Create base map
    m = folium.Map(location=map_center, zoom_start=7, tiles='CartoDB positron')
    
    # Add scale control
    plugins.MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m)
    
    # Set up fleet-specific feature groups if fleet column exists
    fleet_groups = {}
    if 'fleet' in routes_gdf.columns:
        unique_fleets = routes_gdf['fleet'].unique()
        for fleet in unique_fleets:
            if pd.notna(fleet):
                fleet_groups[fleet] = folium.FeatureGroup(name=f"{fleet}")
    
    # Default group for all routes
    all_routes = folium.FeatureGroup(name='All Routes')
    
    # Determine min/max GT for line width scaling
    if 'GT' in routes_gdf.columns:
        min_gt = routes_gdf['GT'].min()
        max_gt = routes_gdf['GT'].max()
    else:
        min_gt = 1000
        max_gt = 100000
    
    # Function to calculate line width based on GT
    def get_line_width(gt):
        if pd.isna(gt) or gt <= 0:
            return 2
        if max_gt == min_gt:
            return 5
        # Logarithmic scale for better visualization
        return 2 + (9 * (gt - min_gt) / (max_gt - min_gt))
    
    # Get colors for fleets
    colors = fleet_colors()
    
    # Add each route to the map
    for idx, route in routes_gdf.iterrows():
        # Get vessel info
        vessel_name = route.get('vessel_name', 'Unknown vessel')
        fleet = route.get('fleet', 'Unknown')
        gt = route.get('GT', 1000)
        
        # Calculate line width
        width = get_line_width(gt)
        
        # Get color based on fleet
        color = colors.get(fleet, '#1E88E5')
        
        # Create popup content
        popup_html = f"""
        <div style="max-width: 300px">
            <h4>{vessel_name}</h4>
            <b>Fleet:</b> {fleet}<br>
            <b>IMO:</b> {route.get('imo', 'Unknown')}<br>
            <b>Gross Tonnage:</b> {int(gt):,} GT<br>
        """
        
        if 'duration_hours' in route:
            popup_html += f"<b>Trip Duration:</b> {route['duration_hours']:.1f} hours<br>"
            
        if 'start_time' in route and 'end_time' in route:
            start = route['start_time'].strftime('%Y-%m-%d %H:%M')
            end = route['end_time'].strftime('%Y-%m-%d %H:%M')
            popup_html += f"<b>Period:</b> {start} to {end}<br>"
            
        popup_html += "</div>"
        
        # Create line coordinates (swap lat/lng for folium)
        line_coords = [(point[1], point[0]) for point in list(route.geometry.coords)]
        
        # Create line object
        line = folium.PolyLine(
            line_coords,
            color=color,
            weight=width,
            opacity=0.7,
            tooltip=f"{vessel_name} ({fleet})",
            popup=folium.Popup(popup_html, max_width=300)
        )
        
        # Add to appropriate group
        if fleet in fleet_groups:
            line.add_to(fleet_groups[fleet])
        else:
            line.add_to(all_routes)
    
    # Add feature groups to map
    for fleet, group in fleet_groups.items():
        group.add_to(m)
    all_routes.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; 
                border: 2px solid grey; z-index: 9999; 
                background-color: white;
                padding: 10px; 
                border-radius: 5px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Fleet Types</div>
    """
    
    # Add fleet colors to legend
    for fleet, color in colors.items():
        if fleet in fleet_groups:
            legend_html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 3px;">
                <span style="display: inline-block; width: 20px; height: 10px; 
                       background-color: {color}; margin-right: 5px;"></span>
                <span>{fleet}</span>
            </div>
            """
    
    legend_html += """
        <div style="margin-top: 8px; font-weight: bold;">Line Width = Gross Tonnage</div>
    </div>
    """
    
    # Add title
    title_html = """
    <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 9999; 
                background-color: white;
                padding: 10px; 
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                font-size: 16px;
                font-weight: bold;">
        Huelva Maritime Traffic - April 2024
    </div>
    """
    
    # Add legend and title to map
    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    m.save(output_file)
    print(f"Map saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate simple maritime route map")
    parser.add_argument('--input', '-i', required=True, help="Input CSV file with processed data")
    parser.add_argument('--output', '-o', default="maritime_map.html", help="Output HTML map file")
    parser.add_argument('--min-speed', type=float, default=0.1, help="Minimum speed threshold")
    parser.add_argument('--min-points', type=int, default=3, help="Minimum points per trip")
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input)
    if df.empty:
        print("No data to process")
        return
    
    # Filter data
    filtered_df = filter_data(df, args.min_speed, args.min_points)
    if filtered_df.empty:
        print("No data left after filtering")
        return
    
    # Create trip lines
    routes_gdf = create_trip_lines(filtered_df)
    if routes_gdf.empty:
        print("No valid routes created")
        return
    
    # Generate map
    create_map(routes_gdf, args.output)
    print("Map generation completed successfully")

if __name__ == "__main__":
    main()
