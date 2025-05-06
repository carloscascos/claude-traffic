#!/usr/bin/env python3
"""
Route Data Preparation Script

This script fetches cruise vessel position data from the coldiron_loc table and prepares a CSV file 
for maritime route analysis. It focuses on a specified time period (default: October 2024).

Usage:
    python route_data_prep.py --year 2024 --month 10 --output "../data/cruise_routes_oct2024.csv"
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from decimal import Decimal

# Import the database module from the project
from database import get_db_connection, execute_query

def fetch_vessel_positions(year, month, limit=None):
    """
    Fetch vessel positions from the coldiron_loc table for a specific time period.
    Join with imo table to get gross tonnage data.
    
    Args:
        year (int): Year of the data to fetch
        month (int): Month of the data to fetch
        limit (int, optional): Maximum number of records to fetch. Defaults to None.
        
    Returns:
        pandas.DataFrame: DataFrame containing vessel positions
    """
    # Build the SQL query
    query = """
    SELECT 
        c.IMO as imo,
        c.lat as latitude,
        c.lng as longitude,
        c.timestamp,
        c.tloc,
        i.GT as GT,
        i.name as vessel_name
    FROM 
        coldiron_loc c
    LEFT OUTER JOIN 
        imo i ON c.IMO = i.IMO
    WHERE 
        YEAR(c.timestamp) = %s 
        AND MONTH(c.timestamp) = %s
    """
    
    params = [year, month]
    
    # Add limit if provided
    if limit:
        query += " ORDER BY c.timestamp LIMIT %s"
        params.append(limit)
    else:
        query += " ORDER BY c.IMO, c.timestamp"
    
    # Execute the query
    print(f"Fetching vessel positions for {month}/{year}...")
    print(f"SQL Query: {query}")
    print(f"Parameters: {params}")
    
    try:
        result = execute_query(query, tuple(params))
        if not result:
            print("No data found for the specified criteria.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result)
        print(f"Retrieved {len(df)} records.")
        
        # Convert Decimal objects to float for GT column if it exists
        if 'GT' in df.columns:
            df['GT'] = df['GT'].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
        
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
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
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(processed_df['timestamp']):
        processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
    
    # Sort by vessel and timestamp
    processed_df = processed_df.sort_values(by=['imo', 'timestamp'])
    
    # Calculate time difference between consecutive positions (for the same vessel)
    processed_df['time_diff'] = processed_df.groupby('imo')['timestamp'].diff().dt.total_seconds()
    
    # Calculate distance between consecutive positions (simple approximation)
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
    
    for vessel, group in processed_df.groupby('imo'):
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
    
    # Add the distance column
    processed_df['distance_km'] = distance_list
    
    # Calculate speed (km/h) based on distance and time difference
    processed_df['calculated_speed'] = processed_df['distance_km'] / (processed_df['time_diff'] / 3600)
    
    # Replace infinite values (division by zero) with NaN
    processed_df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    
    # Create a trip_id column to identify separate voyages (new trip if gap > 2 hours)
    time_threshold = 2 * 60 * 60  # 2 hours in seconds
    
    processed_df['trip_start'] = (
        (processed_df['time_diff'].isna()) |  # First position of a vessel
        (processed_df['time_diff'] > time_threshold)  # Or gap > threshold
    )
    
    # Cumulative sum to create trip IDs
    processed_df['trip_id'] = processed_df.groupby('imo')['trip_start'].cumsum()
    
    # Create a unique trip identifier
    processed_df['unique_trip_id'] = processed_df['imo'].astype(str) + '_' + processed_df['trip_id'].astype(str)
    
    # Drop intermediate columns
    processed_df.drop(['trip_start'], axis=1, inplace=True)
    
    # Fill missing GT with median value per vessel
    if 'GT' in processed_df.columns:
        # Handle GT values that might be strings or other non-numeric types
        processed_df['GT'] = pd.to_numeric(processed_df['GT'], errors='coerce')
        
        # Calculate median GT per vessel (only using valid numeric values)
        median_gt = processed_df.groupby('imo')['GT'].median()
        
        # Fill missing values with the median for that vessel
        overall_median = processed_df['GT'].median()
        
        for imo, group in processed_df.groupby('imo'):
            if group['GT'].isna().any():
                if not pd.isna(median_gt.get(imo)):
                    # Use the vessel's median GT
                    processed_df.loc[processed_df['imo'] == imo, 'GT'] = \
                        processed_df.loc[processed_df['imo'] == imo, 'GT'].fillna(median_gt.get(imo))
                else:
                    # If no GT data for this vessel, use the overall median
                    processed_df.loc[processed_df['imo'] == imo, 'GT'] = \
                        processed_df.loc[processed_df['imo'] == imo, 'GT'].fillna(overall_median)
    
    return processed_df

def create_output_directory(output_path):
    """
    Create the directory for the output file if it doesn't exist.
    
    Args:
        output_path (str): Path to the output file
    """
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

def save_to_csv(df, output_path):
    """
    Save the DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        output_path (str): Path to save the CSV file
    """
    create_output_directory(output_path)
    
    print(f"Saving {len(df)} records to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path}")

def main():
    """
    Main function to fetch and process vessel position data.
    """
    parser = argparse.ArgumentParser(description='Fetch and prepare vessel position data for route analysis.')
    parser.add_argument('--year', type=int, default=2024, help='Year of data to fetch')
    parser.add_argument('--month', type=int, default=10, help='Month of data to fetch')
    parser.add_argument('--limit', type=int, help='Maximum number of records to fetch')
    parser.add_argument('--output', type=str, default='../data/vessel_positions.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Print the parameters
    print(f"Fetching data with parameters:")
    print(f"  Year: {args.year}")
    print(f"  Month: {args.month}")
    print(f"  Limit: {args.limit}")
    print(f"  Output: {args.output}")
    
    # Fetch the data
    df = fetch_vessel_positions(args.year, args.month, args.limit)
    
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
    print(f"  Unique vessels: {processed_df['imo'].nunique()}")
    print(f"  Unique trips: {processed_df['unique_trip_id'].nunique()}")
    print(f"  Date range: {processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}")

if __name__ == "__main__":
    main()
