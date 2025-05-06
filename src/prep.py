#!/usr/bin/env python3
"""
Data Preparation Module

This module extracts and preprocesses raw vessel position data, calculating derived metrics
like speed and distance, and prepares data structures for further analysis.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from decimal import Decimal

def fetch_vessel_positions(input_file=None, limit=None):
    """
    Fetch vessel positions from input file or database.
    
    Args:
        input_file (str, optional): Path to input CSV file. If None, fetch from database
        limit (int, optional): Maximum number of records to fetch. Defaults to None.
        
    Returns:
        pandas.DataFrame: DataFrame containing vessel positions
    """
    if input_file and os.path.exists(input_file):
        print(f"Loading data from {input_file}...")
        try:
            df = pd.read_csv(input_file)
            print(f"Loaded {len(df)} records.")
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return pd.DataFrame()
    else:
        # Database loading would go here
        print("No input file specified or file not found.")
        return pd.DataFrame()

def preprocess_data(df):
    """
    Preprocess the vessel positions data for route analysis.
    
    Args:
        df (pandas.DataFrame): DataFrame with raw vessel positions
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Ensure fleet column exists
    if 'fleet' not in processed_df.columns:
        # Try to derive fleet from vessel type if available
        if 'Type' in processed_df.columns:
            processed_df['fleet'] = processed_df['Type'].apply(categorize_vessel_type)
        elif 'vessel_type' in processed_df.columns:
            processed_df['fleet'] = processed_df['vessel_type'].apply(categorize_vessel_type)
        else:
            # Default to 'Unknown' if no type information is available
            processed_df['fleet'] = 'Unknown'
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in processed_df.columns and not pd.api.types.is_datetime64_any_dtype(processed_df['timestamp']):
        processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
    
    # Sort by vessel and timestamp
    if 'imo' in processed_df.columns and 'timestamp' in processed_df.columns:
        processed_df = processed_df.sort_values(by=['imo', 'timestamp'])
    
    # Calculate time difference between consecutive positions (for the same vessel)
    if 'imo' in processed_df.columns and 'timestamp' in processed_df.columns:
        processed_df['time_diff'] = processed_df.groupby('imo')['timestamp'].diff().dt.total_seconds()
    
    # Calculate distance between consecutive positions
    if 'latitude' in processed_df.columns and 'longitude' in processed_df.columns:
        # Calculate distance for consecutive points of the same vessel
        distance_list = calculate_distances(processed_df)
        processed_df['distance_km'] = distance_list
    
    # Calculate speed based on distance and time difference
    if 'distance_km' in processed_df.columns and 'time_diff' in processed_df.columns:
        processed_df['calculated_speed'] = processed_df['distance_km'] / (processed_df['time_diff'] / 3600)
        # Replace infinite values (division by zero) with NaN
        processed_df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    
    # Create a trip_id column to identify separate voyages (new trip if gap > 2 hours)
    if 'time_diff' in processed_df.columns:
        time_threshold = 2 * 60 * 60  # 2 hours in seconds
        processed_df['trip_start'] = (
            (processed_df['time_diff'].isna()) |  # First position of a vessel
            (processed_df['time_diff'] > time_threshold)  # Or gap > threshold
        )
        # Cumulative sum to create trip IDs
        if 'imo' in processed_df.columns:
            processed_df['trip_id'] = processed_df.groupby('imo')['trip_start'].cumsum()
            # Create a unique trip identifier
            processed_df['unique_trip_id'] = processed_df['imo'].astype(str) + '_' + processed_df['trip_id'].astype(str)
            # Drop intermediate columns
            processed_df.drop(['trip_start'], axis=1, inplace=True)
    
    return processed_df

def calculate_distances(df):
    """
    Calculate distances between consecutive points for each vessel.
    
    Args:
        df (pandas.DataFrame): DataFrame with vessel positions
        
    Returns:
        list: List of distances in kilometers
    """
    # Function to calculate Haversine distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        from math import radians, cos, sin, asin, sqrt
        
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    # Calculate distance for consecutive points of the same vessel
    distance_list = []
    
    for vessel, group in df.groupby('imo'):
        # Reset index for easier iteration
        group = group.reset_index(drop=True)
        
        # First point has no previous point, so distance is NaN
        distances = [float('nan')]
        
        # Calculate distances for remaining points
        for i in range(1, len(group)):
            prev_lat = group.iloc[i-1]['latitude']
            prev_lon = group.iloc[i-1]['longitude']
            curr_lat = group.iloc[i]['latitude']
            curr_lon = group.iloc[i]['longitude']
            
            dist = haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)
            distances.append(dist)
        
        # Append the distances for this vessel to the main list
        distance_list.extend(distances)
    
    return distance_list

def categorize_vessel_type(vessel_type):
    """
    Categorize vessel types into fleet categories.
    
    Args:
        vessel_type (str): Original vessel type
        
    Returns:
        str: Fleet category
    """
    if pd.isna(vessel_type) or not vessel_type:
        return 'Unknown'
    
    vessel_type = str(vessel_type).lower()
    
    # Define fleet categories based on vessel type keywords
    if any(keyword in vessel_type for keyword in ['cruise', 'passenger']):
        return 'Cruise/Passenger'
    elif any(keyword in vessel_type for keyword in ['container']):
        return 'Container'
    elif any(keyword in vessel_type for keyword in ['bulk', 'cargo']):
        return 'Bulk Carrier'
    elif any(keyword in vessel_type for keyword in ['tanker', 'oil', 'chemical']):
        return 'Tanker'
    elif any(keyword in vessel_type for keyword in ['ferry', 'ro-ro', 'roro']):
        return 'Ferry/Ro-Ro'
    elif any(keyword in vessel_type for keyword in ['fishing']):
        return 'Fishing'
    elif any(keyword in vessel_type for keyword in ['yacht', 'pleasure']):
        return 'Pleasure'
    elif any(keyword in vessel_type for keyword in ['tug']):
        return 'Tug'
    else:
        return 'Other'

def save_to_csv(df, output_path):
    """
    Save the DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        output_path (str): Path to save the CSV file
    """
    # Create output directory if it doesn't exist
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)
    
    print(f"Saving {len(df)} records to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path}")

def main():
    """
    Main function to fetch and process vessel position data.
    """
    parser = argparse.ArgumentParser(description='Fetch and prepare vessel position data for route analysis.')
    parser.add_argument('--input', type=str, help='Input CSV file path (if not using database)')
    parser.add_argument('--output', type=str, default='data/processed_data.csv', help='Output CSV file path')
    parser.add_argument('--limit', type=int, help='Maximum number of records to fetch')
    
    args = parser.parse_args()
    
    # Print the parameters
    print(f"Running data preparation with parameters:")
    print(f"  Input: {args.input if args.input else 'Using database'}")
    print(f"  Output: {args.output}")
    print(f"  Limit: {args.limit}")
    
    # Fetch the data
    df = fetch_vessel_positions(args.input, args.limit)
    
    if df.empty:
        print("No data to process. Exiting.")
        return
    
    # Process the data
    processed_df = preprocess_data(df)
    
    # Save to CSV
    save_to_csv(processed_df, args.output)
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"  Total positions: {len(processed_df)}")
    if 'imo' in processed_df.columns:
        print(f"  Unique vessels: {processed_df['imo'].nunique()}")
    if 'unique_trip_id' in processed_df.columns:
        print(f"  Unique trips: {processed_df['unique_trip_id'].nunique()}")
    if 'timestamp' in processed_df.columns:
        print(f"  Date range: {processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}")
    if 'fleet' in processed_df.columns:
        print("\nFleet Distribution:")
        fleet_counts = processed_df['fleet'].value_counts()
        for fleet, count in fleet_counts.items():
            print(f"  {fleet}: {count} positions ({count/len(processed_df)*100:.1f}%)")

if __name__ == "__main__":
    main()
