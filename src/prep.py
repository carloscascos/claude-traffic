#!/usr/bin/env python3
"""
Maritime Route Data Preparation Script

This script fetches vessel position data from the database and prepares a CSV file 
for maritime route analysis. It focuses on a specified time period and can extract
data for specific vessel types.

Usage:
    python prep.py --year 2024 --month 10 --output "data/vessel_routes_oct2024.csv"
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from decimal import Decimal
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """
    Establishes a connection with the database using environment variables.
    
    Returns:
        mysql.connector.connection: Database connection object
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "imo")
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

def execute_query(query, params=None):
    """
    Execute a SQL query in the database.
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters for the query. Defaults to None.
        
    Returns:
        list: Results of the query or None if error
    """
    connection = get_db_connection()
    if not connection:
        return None
    
    try:
        cursor = connection.cursor(dictionary=True)
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def fetch_vessel_positions(year, month, vessel_type=None, limit=None):
    """
    Fetch vessel positions from the database for a specific time period.
    Join with imo table to get vessel information.
    
    Args:
        year (int): Year of the data to fetch
        month (int): Month of the data to fetch
        vessel_type (str, optional): Type of vessel to filter by. Defaults to None.
        limit (int, optional): Maximum number of records to fetch. Defaults to None.
        
    Returns:
        pandas.DataFrame: DataFrame containing vessel positions
    """
    # Build the SQL query
    query = """
    SELECT 
        v.imo as imo,
        v.lat as latitude,
        v.lng as longitude,
        v.timestamp,
        v.tloc,
        v.speed,
        i.GT as GT,
        i.NAME as vessel_name,
        i.Type as vessel_type,
        i.FLAG as flag
    FROM 
        v_loc v
    LEFT OUTER JOIN 
        imo i ON v.imo = i.IMO
    WHERE 
        YEAR(v.timestamp) = %s 
        AND MONTH(v.timestamp) = %s
    """
    
    params = [year, month]
    
    # Add vessel type filter if provided
    if vessel_type:
        query += " AND i.Type = %s"
        params.append(vessel_type)
    
    # Add limit if provided
    if limit:
        query += " ORDER BY v.timestamp LIMIT %s"
        params.append(limit)
    else:
        query += " ORDER BY v.imo, v.timestamp"
    
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
        
        # Convert speed to float (it might be stored as text in the database)
        if 'speed' in df.columns:
            df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

def fetch_port_based_vessel_positions(port_name, start_date, end_date, window_days=7, limit=None):
    """
    Fetch vessel positions for vessels that called at a specific port within a date range.
    
    Args:
        port_name (str): Name of the port
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        window_days (int, optional): Number of days before/after port call to include. Defaults to 7.
        limit (int, optional): Maximum number of records to fetch. Defaults to None.
        
    Returns:
        pandas.DataFrame: DataFrame containing vessel positions
    """
    # First, find vessels that called at the port during the date range
    port_query = """
    SELECT 
        imo, 
        start, 
        end
    FROM 
        escalas
    WHERE 
        portname = %s
        AND start BETWEEN %s AND %s
    """
    
    port_params = [port_name, start_date, end_date]
    
    print(f"Finding vessels that called at {port_name} between {start_date} and {end_date}...")
    
    port_calls = execute_query(port_query, tuple(port_params))
    
    if not port_calls:
        print(f"No vessels found calling at {port_name} during the specified period.")
        return pd.DataFrame()
    
    # Extract unique IMOs
    imos = [call['imo'] for call in port_calls]
    unique_imos = list(set(imos))
    
    print(f"Found {len(port_calls)} port calls by {len(unique_imos)} unique vessels.")
    
    # For each vessel, get positions around the port call time
    all_positions = []
    
    for call in port_calls:
        imo = call['imo']
        call_start = call['start']
        call_end = call['end']
        
        # Calculate the time window
        window_start = (call_start - timedelta(days=window_days)).strftime('%Y-%m-%d')
        window_end = (call_end + timedelta(days=window_days)).strftime('%Y-%m-%d')
        
        # Query for vessel positions
        positions_query = """
        SELECT 
            v.imo as imo,
            v.lat as latitude,
            v.lng as longitude,
            v.timestamp,
            v.tloc,
            v.speed,
            i.GT as GT,
            i.NAME as vessel_name,
            i.Type as vessel_type,
            i.FLAG as flag
        FROM 
            v_loc v
        LEFT OUTER JOIN 
            imo i ON v.imo = i.IMO
        WHERE 
            v.imo = %s
            AND v.timestamp BETWEEN %s AND %s
        ORDER BY 
            v.timestamp
        """
        
        positions_params = [imo, window_start, window_end]
        
        if limit:
            positions_query += " LIMIT %s"
            positions_params.append(limit)
        
        vessel_positions = execute_query(positions_query, tuple(positions_params))
        
        if vessel_positions:
            all_positions.extend(vessel_positions)
    
    if not all_positions:
        print("No vessel positions found for the port calls.")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_positions)
    print(f"Retrieved {len(df)} position records for vessels calling at {port_name}.")
    
    # Convert Decimal objects to float for GT column if it exists
    if 'GT' in df.columns:
        df['GT'] = df['GT'].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
    
    # Convert speed to float (it might be stored as text in the database)
    if 'speed' in df.columns:
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    
    return df

def fetch_full_year_data(port_name, year, window_days=7, max_vessels=None):
    """
    Fetch vessel positions for a full year in monthly batches.
    
    Args:
        port_name (str): Name of the port
        year (int): Year to fetch data for
        window_days (int, optional): Number of days before/after port call to include. Defaults to 7.
        max_vessels (int, optional): Maximum number of vessels to process. Defaults to None.
        
    Returns:
        pandas.DataFrame: DataFrame containing vessel positions
    """
    all_data = []
    
    # Process each month
    for month in range(1, 13):
        start_date = f"{year}-{month:02d}-01"
        
        # Calculate end date (last day of month)
        if month == 12:
            end_date = f"{year}-12-31"
        else:
            end_date = f"{year}-{month+1:02d}-01"
            # Convert to datetime, subtract 1 day, convert back to string
            end_date = (pd.to_datetime(end_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"Processing month {month}: {start_date} to {end_date}")
        
        # Get data for this month
        month_data = fetch_port_based_vessel_positions(
            port_name, 
            start_date, 
            end_date, 
            window_days
        )
        
        if not month_data.empty:
            all_data.append(month_data)
            
            # Check if we've reached max vessels
            if max_vessels and len(set(month_data['imo'])) >= max_vessels:
                print(f"Reached maximum vessel limit ({max_vessels}). Stopping data collection.")
                break
    
    # Combine all months
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Remove duplicates (same IMO and tloc)
        combined_data = combined_data.drop_duplicates(subset=['imo', 'tloc'])
        return combined_data
    else:
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
        r = 3440.065  # Radius of earth in nautical miles
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
    processed_df['distance_nm'] = distance_list
    
    # Calculate speed (knots) based on distance and time difference
    # Note: 1 knot = 1 nautical mile per hour
    processed_df['calculated_speed'] = processed_df['distance_nm'] / (processed_df['time_diff'] / 3600)
    
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
    parser.add_argument('--port', type=str, help='Port name to filter by (if using port-based extraction)')
    parser.add_argument('--vessel-type', type=str, help='Vessel type to filter by')
    parser.add_argument('--window-days', type=int, default=7, 
                        help='Number of days before/after port call to include (for port-based extraction)')
    parser.add_argument('--start-date', type=str, help='Start date for port-based extraction (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for port-based extraction (YYYY-MM-DD)')
    parser.add_argument('--full-year', action='store_true', help='Process a full year of data in monthly batches')
    parser.add_argument('--max-vessels', type=int, help='Maximum number of vessels to process (for full-year extraction)')
    parser.add_argument('--limit', type=int, help='Maximum number of records to fetch')
    parser.add_argument('--output', type=str, default='data/vessel_positions.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Print the parameters
    print(f"Fetching data with parameters:")
    for arg in vars(args):
        if getattr(args, arg) is not None:
            print(f"  {arg}: {getattr(args, arg)}")
    
    # Fetch the data based on the extraction method
    if args.full_year and args.port:
        # Full year extraction for a specific port
        df = fetch_full_year_data(args.port, args.year, args.window_days, args.max_vessels)
    elif args.port and (args.start_date and args.end_date):
        # Port-based extraction for a specific date range
        df = fetch_port_based_vessel_positions(
            args.port, 
            args.start_date, 
            args.end_date, 
            args.window_days,
            args.limit
        )
    else:
        # General extraction for a specific month
        df = fetch_vessel_positions(args.year, args.month, args.vessel_type, args.limit)
    
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
    if not processed_df['timestamp'].empty:
        print(f"  Date range: {processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}")

if __name__ == "__main__":
    main()
