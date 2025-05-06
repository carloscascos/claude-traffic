#!/usr/bin/env python3
"""
Maritime Route Data Preparation Script

This script fetches vessel position data based on port calls, retrieving data for
N days before and N days after each port call. It handles duplicate positions 
that may occur when vessels call at the same port multiple times.

Usage:
    python prep.py --port "Barcelona" --year 2024 --month 10 
                  --window-days 7 --output "data/barcelona_oct2024.csv"
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import asyncio
import concurrent.futures
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

def find_port_calls(port_name, start_date, end_date):
    """
    Find all vessels that called at a specific port within a date range.
    
    Args:
        port_name (str): Name of the port
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        list: List of dictionaries with port call information
    """
    port_query = """
    SELECT 
        imo, 
        start, 
        end,
        portname,
        country,
        prev_port,
        next_port
    FROM 
        escalas
    WHERE 
        portname = %s
        AND start BETWEEN %s AND %s
    ORDER BY
        imo, start
    """
    
    port_params = [port_name, start_date, end_date]
    
    print(f"Finding vessels that called at {port_name} between {start_date} and {end_date}...")
    
    port_calls = execute_query(port_query, tuple(port_params))
    
    if not port_calls:
        print(f"No vessels found calling at {port_name} during the specified period.")
        return []
    
    # Extract unique IMOs
    unique_imos = list(set(call['imo'] for call in port_calls))
    
    print(f"Found {len(port_calls)} port calls by {len(unique_imos)} unique vessels.")
    
    return port_calls

async def fetch_vessel_positions_for_call(call, window_days, max_positions=None, semaphore=None):
    """
    Fetch vessel positions for a single port call with window days before and after.
    
    Args:
        call (dict): Port call information
        window_days (int): Number of days before/after port call to include
        max_positions (int, optional): Maximum number of positions to fetch per call
        semaphore (asyncio.Semaphore, optional): Semaphore for limiting concurrent queries
        
    Returns:
        pandas.DataFrame: DataFrame with vessel positions for this call
    """
    imo = call['imo']
    call_start = call['start']
    call_end = call['end']
    port_name = call['portname']
    
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
        v.fleet as fleet,
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
    
    if max_positions:
        positions_query += " LIMIT %s"
        positions_params.append(max_positions)
    
    try:
        # Use semaphore if provided to limit concurrent queries
        if semaphore:
            async with semaphore:
                # Run the database query in a thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                vessel_positions = await loop.run_in_executor(
                    None, 
                    lambda: execute_query(positions_query, tuple(positions_params))
                )
        else:
            vessel_positions = execute_query(positions_query, tuple(positions_params))
        
        if not vessel_positions:
            print(f"No positions found for vessel IMO {imo} around port call at {port_name}.")
            return pd.DataFrame()
        
        # Add port call information to each position record
        for position in vessel_positions:
            position['port_call_start'] = call_start
            position['port_call_end'] = call_end
            position['port_name'] = port_name
            position['call_country'] = call.get('country')
            position['prev_port'] = call.get('prev_port')
            position['next_port'] = call.get('next_port')
        
        # Convert to DataFrame
        df = pd.DataFrame(vessel_positions)
        
        # Convert Decimal objects to float for GT column if it exists
        if 'GT' in df.columns:
            df['GT'] = df['GT'].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
        
        # Convert speed to float (it might be stored as text in the database)
        if 'speed' in df.columns:
            df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"Error fetching positions for vessel IMO {imo}: {e}")
        return pd.DataFrame()

async def process_port_calls(port_calls, window_days, max_positions=None, max_concurrency=5):
    """
    Process all port calls in parallel, collecting vessel positions for each.
    
    Args:
        port_calls (list): List of port call dictionaries
        window_days (int): Number of days before/after port call to include
        max_positions (int, optional): Maximum number of positions to fetch per call
        max_concurrency (int, optional): Maximum number of concurrent queries. Defaults to 5.
        
    Returns:
        pandas.DataFrame: Combined DataFrame with all vessel positions
    """
    print(f"Processing {len(port_calls)} port calls with window of {window_days} days before/after...")
    print(f"Maximum concurrency: {max_concurrency} vessels at a time")
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # Create tasks for each port call
    tasks = [
        fetch_vessel_positions_for_call(call, window_days, max_positions, semaphore)
        for call in port_calls
    ]
    
    # Execute all tasks and gather results
    all_results = await asyncio.gather(*tasks)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Remove duplicate records (same IMO and timestamp)
    original_size = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['imo', 'timestamp'])
    duplicates_removed = original_size - len(combined_df)
    
    print(f"Combined {len(all_results)} port calls into {len(combined_df)} position records.")
    print(f"Removed {duplicates_removed} duplicate positions ({duplicates_removed/original_size*100:.1f}% of total).")
    
    return combined_df

def process_data_for_month(port_name, year, month, window_days, max_positions=None, max_concurrency=5):
    """
    Process port calls for a specific month.
    
    Args:
        port_name (str): Name of the port
        year (int): Year to process
        month (int): Month to process
        window_days (int): Number of days before/after port call to include
        max_positions (int, optional): Maximum number of positions to fetch per call
        max_concurrency (int, optional): Maximum number of concurrent queries
        
    Returns:
        pandas.DataFrame: DataFrame with vessel positions
    """
    # Calculate start and end date for the month
    start_date = f"{year}-{month:02d}-01"
    
    # Calculate end date (last day of month)
    if month == 12:
        end_date = f"{year}-12-31"
    else:
        end_date = f"{year}-{month+1:02d}-01"
        # Convert to datetime, subtract 1 day, convert back to string
        end_date = (pd.to_datetime(end_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Processing port calls for {port_name} in {year}-{month:02d}...")
    
    # Find port calls for this month
    port_calls = find_port_calls(port_name, start_date, end_date)
    
    if not port_calls:
        print(f"No port calls found for {port_name} in {year}-{month:02d}.")
        return pd.DataFrame()
    
    # Process port calls
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        combined_df = loop.run_until_complete(
            process_port_calls(port_calls, window_days, max_positions, max_concurrency)
        )
    finally:
        loop.close()
    
    return combined_df

def process_full_year(port_name, year, window_days, max_positions=None, max_concurrency=5, max_vessels=None):
    """
    Process port calls for a full year, month by month.
    
    Args:
        port_name (str): Name of the port
        year (int): Year to process
        window_days (int): Number of days before/after port call to include
        max_positions (int, optional): Maximum number of positions to fetch per call
        max_concurrency (int, optional): Maximum number of concurrent queries
        max_vessels (int, optional): Maximum number of vessels to process
        
    Returns:
        pandas.DataFrame: DataFrame with vessel positions
    """
    print(f"Processing full year {year} for port {port_name}...")
    
    all_data = []
    vessels_processed = set()
    
    # Process each month
    for month in range(1, 13):
        print(f"Processing month {month}...")
        
        month_data = process_data_for_month(
            port_name, year, month, window_days, max_positions, max_concurrency
        )
        
        if not month_data.empty:
            # Add to results
            all_data.append(month_data)
            
            # Track vessels processed
            vessels_processed.update(month_data['imo'].unique())
            
            # Check if we've reached max vessels
            if max_vessels and len(vessels_processed) >= max_vessels:
                print(f"Reached maximum vessel limit ({max_vessels}). Stopping data collection.")
                break
    
    # Combine all months
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Final deduplication in case there's overlap between months
        combined_data = combined_data.drop_duplicates(subset=['imo', 'timestamp'])
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
    parser.add_argument('--port', type=str, required=True, help='Port name to filter by')
    parser.add_argument('--year', type=int, default=2024, help='Year of data to fetch')
    parser.add_argument('--month', type=int, help='Month of data to fetch (if processing a single month)')
    parser.add_argument('--window-days', type=int, default=7, 
                        help='Number of days before/after port call to include')
    parser.add_argument('--start-date', type=str, help='Custom start date for port calls (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Custom end date for port calls (YYYY-MM-DD)')
    parser.add_argument('--full-year', action='store_true', help='Process a full year of data in monthly batches')
    parser.add_argument('--max-positions', type=int, help='Maximum number of positions to fetch per call')
    parser.add_argument('--max-vessels', type=int, help='Maximum number of vessels to process')
    parser.add_argument('--max-concurrency', type=int, default=5, 
                        help='Maximum number of concurrent vessel queries')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--debug', action='store_true', help='Enable debugging output')
    
    args = parser.parse_args()
    
    # Print the parameters
    print(f"Fetching data with parameters:")
    for arg in vars(args):
        if getattr(args, arg) is not None:
            print(f"  {arg}: {getattr(args, arg)}")
    
    # Get vessel positions based on the extraction method
    if args.full_year:
        # Process full year
        df = process_full_year(
            args.port, 
            args.year, 
            args.window_days, 
            args.max_positions,
            args.max_concurrency,
            args.max_vessels
        )
    elif args.month:
        # Process specific month
        df = process_data_for_month(
            args.port, 
            args.year, 
            args.month, 
            args.window_days, 
            args.max_positions,
            args.max_concurrency
        )
    elif args.start_date and args.end_date:
        # Process custom date range
        port_calls = find_port_calls(args.port, args.start_date, args.end_date)
        
        if not port_calls:
            print("No port calls found for the specified criteria. Exiting.")
            return
        
        # Process port calls
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            df = loop.run_until_complete(
                process_port_calls(
                    port_calls, 
                    args.window_days, 
                    args.max_positions, 
                    args.max_concurrency
                )
            )
        finally:
            loop.close()
    else:
        print("Error: You must specify either --month, --full-year, or both --start-date and --end-date")
        return
    
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
    if not processed_df.empty and 'timestamp' in processed_df:
        print(f"  Date range: {processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}")

if __name__ == "__main__":
    main()
