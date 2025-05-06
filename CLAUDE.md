# Claude-Traffic Project Overview

This project is located in /home/ccascos/proyectos/claude-traffic (carloscascos/claude-traffic in github) 

you are running in a WLS (Windows Linux Subsystem) with the following structure, this means that all  yoour mco runs in a linux file systemn

desktop-commander mcp is restricted to write under /home/ccascos/proyectos/claude-traffic only never out of this folder




when ask you to "save" or "write" code, you should save it in the project folder
when ask you to "run" code, you should "save" it if needed and run it in the project folder

make sure "env" python envvironment is activated before running any code as "source ./env/bin/activate && python your_script.py"


# after reading this file

Sync the status of the actual user project folder with CLAUDE context so we start always with real data from the user project folder
wait for instructions from the user


## permissions

before writing code ALWAYS ask the user with 4 options

1/ yes for this time
2/ yes for all session
3/ no, just summarize what would you do





## Project Purpose

The Claude-Traffic project is a maritime data processing and visualization system designed to analyze vessel traffic patterns and create interactive "maritime highway" maps. The system ingests AIS vessel position data and transforms it into insightful visualizations that show common shipping routes with line widths proportional to traffic volume (measured by gross tonnage).

## Current Project Status (Updated May 2025)

The project is now fully functional with several key improvements:

1. **Enhanced Port-Based Extraction**: The Barcelona port extraction script has been optimized to:
   - Process vessels with multiple calls at the same port only once (using MIN/MAX dates)
   - Avoid SQLAlchemy warnings by using direct cursor-based database queries
   - Limit concurrent operations to 5 vessels at a time for stability
   - Improved date format handling for more reliable data extraction

2. **Improved Data Processing**:
   - Fixed error handling for missing or null values in database queries
   - More robust handling of speed data (stored as text in the database)
   - Better date format conversion for various input formats

3. **Performance Optimizations**:
   - Added asyncio semaphore to control concurrency
   - More memory-efficient processing of large datasets
   - Better progress logging and debugging options

4. **New Full-Year Data Extraction**:
   - Added `barcelona_full2024_prep.py` for processing a full year of vessel data
   - Implements batch processing to handle large time ranges without memory issues
   - Uses monthly batches for efficient database querying
   - Successfully processes thousands of vessel position records across multiple vessels

5. **Enhanced Visualizations**:
   - Improved `maritime_routes_by_type.py` script that colors routes by vessel type
   - Optimized DBSCAN parameters to better identify distinct route patterns
   - Lower minimum samples (2) to capture routes with fewer data points
   - Smaller epsilon values (0.004-0.005) to distinguish closely parallel routes
   - Shows individual routes when not enough positions can be clustered

The latest enhancements focus on route differentiation and visualization quality, presenting maritime traffic patterns in a more informative and visually appealing way.

## Core Architecture

The project follows a decoupled, modular architecture with two primary components:

1. **Data Extraction and Preprocessing**: Multiple strategies for preparing vessel position data
2. **Route Analysis and Visualization**: A standardized method for analyzing and visualizing maritime routes

### Key Design Principles

- **Decoupling**: The data preparation phase is fully decoupled from the visualization phase
- **Standardized Interface**: A well-defined CSV format connects the two components
- **Multiple Preparation Strategies**: Support for different data extraction approaches while maintaining consistent output
- **Maritime Standards**: Consistent use of maritime units (nautical miles, knots) and terminology

## Standardized Data Interface (CSV Format)

All data preparation methods must produce CSV files with the following standardized format:

**Required Columns:**
- `imo`: Integer - Vessel IMO number
- `vessel_name`: String - Name of the vessel
- `timestamp`: Datetime - Timestamp of the position
- `lat`: Float - Latitude coordinate
- `lng`: Float - Longitude coordinate
- `GT`: Float - Gross tonnage
- `speed`: Float - Speed in knots
- `unique_trip_id`: String - Identifier for the trip (format: "imo_tripnumber")

**Additional Columns:**
- `tloc`: String - AIS transmission location code (used as part of primary key with IMO)
- `trip_id`: Integer - Trip segment identifier
- `distance_nm`: Float - Distance from previous point in nautical miles
- `time_diff`: Float - Time difference from previous point (seconds)
- `prev_leg`: Float - Previous leg distance in nautical miles
- `next_leg`: Float - Next leg distance in nautical miles

## Data Preparation Strategies

We've implemented multiple strategies for data preparation:

1. **General Purpose Extractor** (`route_data_prep.py`):
   - Fetches vessel positions from the coldiron_loc table for a specific time period
   - Joins with imo table to get vessel information
   - Calculates time differences, distances, and speeds
   - Groups positions into discrete trips based on time gaps

2. **Port-Based Extractors**:
   - **Monthly Data** (`barcelona_jan2025_prep.py`):
     - Focuses on vessels calling at specific ports
     - Extracts vessel routes within a time window (e.g., 7 days before/after port call)
     - Uses asyncio for parallel processing of multiple vessels
     - Produces the same standardized CSV format
     - Optimized to handle multiple port calls efficiently

   - **Full-Year Data** (`barcelona_full2024_prep.py`):
     - Processes an entire year of port calls in monthly batches
     - Uses controlled concurrency for parallel vessel processing
     - Efficiently handles date ranges spanning many months
     - Memory-efficient processing of large datasets
     - Creates detailed CSV output suitable for sophisticated visualizations

3. **Batch Processing Script** (`run-maritime-analysis.sh`):
   - Pipeline to process multiple months of data
   - Combines data into a comprehensive dataset

## Visualization Components

The visualization components take any CSV file in the standardized format and generate maritime highway maps:

1. **Basic Route Visualizer** (`cruise_highway_map.py` / `cruise_highway_map_fixed.py`):
   - Uses DBSCAN clustering to identify common routes
   - Aggregates similar routes and calculates total gross tonnage
   - Generates interactive HTML maps with Folium
   - Line widths proportional to vessel traffic/gross tonnage
   - Automatically identifies port locations from vessel stopping patterns

2. **Advanced Routes by Vessel Type** (`maritime_routes_by_type.py`):
   - Colors routes based on vessel type (container ships, tankers, bulk carriers, etc.)
   - Uses optimized DBSCAN parameters for better route differentiation
   - Shows both clustered route patterns and individual routes
   - Interactive toggles for different vessel types
   - Detailed popup information for each route
   - More refined visualization suitable for professional analysis

3. **Simplified Visualizer** (`simple_map.py`):
   - Creates basic route visualizations without complex clustering
   - Useful for quick visualization or smaller datasets

## DBSCAN Parameter Optimization

Through experimentation, we've found the optimal DBSCAN parameters for maritime route clustering:

1. **Epsilon (eps)**:
   - Default cluster_eps: 0.01-0.1 (for broader clusters)
   - Fine-grained routes: 0.003-0.005 (for detailed route differentiation)
   - Values need adjustment based on the density of AIS data

2. **Minimum Samples (min_samples)**:
   - Setting to 2 allows identification of less common routes
   - Higher values (3-5) focus on more heavily traveled routes

3. **Route Visualization**:
   - `--routes_only` parameter removes port markers and other distractions
   - Color-coding by vessel type significantly improves interpretability
   - Showing less common routes gives a more complete picture of maritime traffic

## Async Data Processing Framework

The system leverages Python's async capabilities for parallel processing:

- Multiple ports can be processed simultaneously
- Multiple vessels can be processed in parallel (now limited to 5 by default)
- Uses ProcessPoolExecutor for CPU-bound tasks
- Standardized output regardless of the data source
- Batch processing for full-year analysis

## Full-Year Data Processing Example

The `barcelona_full2024_prep.py` script demonstrates a strategy for processing an entire year of data:

1. Queries all unique vessels calling at Barcelona throughout 2024 in monthly batches
2. For each vessel, processes positions in 30-day segments to avoid memory issues
3. Coordinates multiple concurrent vessel processing tasks with semaphores
4. Uses process pools to handle CPU-intensive calculations
5. Combines data with proper deduplication
6. Generates standardized CSV ready for advanced visualization

### Usage Example for Full-Year Analysis

```bash
# Basic usage
python ./src/barcelona_full2024_prep.py --output ./data/barcelona_full2024_routes.csv

# With limited vessel processing for testing
python ./src/barcelona_full2024_prep.py --output ./data/barcelona_full2024_routes.csv --max-vessels 20 --debug

# Custom parameters
python ./src/barcelona_full2024_prep.py --port "Valencia" --start 2024-01-01 --end 2024-12-31 --window 10 --max-concurrency 3 --output ./data/valencia_2024_routes.csv
```

## Routes by Vessel Type Visualization

The `maritime_routes_by_type.py` script creates sophisticated visualizations:

1. Loads vessel data from standardized CSV format
2. Fetches vessel type information from the database
3. Clusters routes separately for each vessel type
4. Color-codes routes based on vessel category
5. Uses optimized DBSCAN parameters for better differentiation of maritime corridors
6. Generates interactive HTML maps with toggleable layers for each vessel type

### Usage Example for Route by Type Visualization

```bash
# Basic usage with optimized parameters
python ./src/maritime_routes_by_type.py --input ./data/barcelona_full2024_routes.csv --output ./results/barcelona_routes_by_vessel_type.html --cluster_eps 0.004 --cluster_min_samples 2 --min_trip_points 5 --routes_only

# For broader route clusters
python ./src/maritime_routes_by_type.py --input ./data/barcelona_full2024_routes.csv --output ./results/barcelona_major_routes.html --cluster_eps 0.02 --cluster_min_samples 3
```

## Database Schema

The system works with the following key tables:

- `v_loc`: Vessel position data view with the following columns:
  - imo (int): Vessel IMO number
  - lat/lng (double): Position coordinates  
  - timestamp (timestamp): Time of position report
  - tloc (timestamp): Transmission location timestamp
  - speed (mediumtext): Speed data (stored as text)
  - fleet (varchar): Fleet identifier
  - GT (decimal): Gross tonnage

- `imo`: Vessel information (name, gross tonnage, type)
- `escalas`: Port calls data (start, end, port name, IMO)

### Fleet Categories in v_loc

The `v_loc` view combines multiple vessel type tables, with the `fleet` column containing specific standardized values that identify vessel categories:

1. `lngtankers` - LNG (Liquefied Natural Gas) tanker vessels
2. `lpgtankers` - LPG (Liquefied Petroleum Gas) tanker vessels
3. `containers` - Container ships
4. `carcarriers` - Car carrier vessels
5. `bulkcarriers` - Bulk carrier vessels
6. `ferrys` - Ferry vessels
7. `cruises` - Cruise ships
8. `oiltankers` - Oil tanker vessels

These fleet values should be used directly for route visualization color-coding to maintain consistency with the database structure. When querying the `v_loc` view, the `fleet` column already provides properly categorized vessel types without requiring additional lookups to the `imo` table.

## Future Development Directions

1. **Additional Data Preparation Strategies**:
   - Fleet-based analysis (tracking specific vessel fleets)
   - Grid-based analysis (focusing on specific geographic regions)
   - Weather-integrated analysis (correlating routes with weather patterns)

2. **Enhanced Visualization Features**:
   - Time-based animations of traffic patterns
   - Integration with additional data layers (weather, port congestion)
   - Predictive route modeling
   - Interactive traffic density heatmaps

3. **Performance Optimizations**:
   - Database query optimizations for larger datasets
   - Distributed processing for very large fleets
   - Incremental data updates

4. **More Advanced Route Analysis**:
   - Machine learning for route prediction
   - Anomaly detection for unusual vessel behavior
   - Seasonal traffic pattern analysis
   - Integration with vessel draft and cargo type data

## Project Organization

```
claude-traffic/
├── data/               # Stores processed CSV files
├── docs/               # Additional project documentation
├── src/                # Source code
│   ├── database.py     # Database connection utilities
│   ├── route_data_prep.py  # General purpose data preparation
│   ├── barcelona_jan2025_prep.py  # Port-based data preparation (monthly)
│   ├── barcelona_full2024_prep.py  # Full-year data preparation
│   ├── cruise_highway_map.py  # Basic route visualization with DBSCAN
│   ├── cruise_highway_map_fixed.py  # Fixed version of route visualization
│   ├── maritime_routes_by_type.py  # Advanced visualization by vessel type
│   ├── simple_map.py   # Simplified visualization
│   ├── visualization.py  # Visualization utilities
│   ├── analysis.py     # Data analysis utilities
│   ├── main.py         # CLI interface
│   └── run-maritime-analysis.sh  # Batch processing script
├── results/            # Output visualization HTML files
├── .env                # Environment variables (not in version control)
├── .gitignore          # Git ignore file
├── README.md           # Project overview
├── requirements.txt    # Python dependencies
└── CLAUDE.md           # This file - comprehensive project context
```

## Key Technical Decisions

1. **DBSCAN for Route Clustering**:
   - Effective for identifying similar maritime routes despite variations
   - Parameters can be tuned based on desired clustering granularity (eps=0.004-0.1)
   - Minimum samples set to 2-3 for optimal route identification
   - Separate clustering by vessel type improves results

2. **Folium for Interactive Maps**:
   - Web-based interactive visualization
   - Supports multiple data layers and popup information
   - Color-coding by vessel type with customizable legends
   - Layer toggling for different vessel categories

3. **Asyncio for Parallel Processing**:
   - Efficiently processes multiple vessels/ports simultaneously
   - Dramatically improves performance for large datasets
   - Now with controlled concurrency (5 vessels max by default)
   - Batch processing for full-year data analysis

4. **Standardized CSV Interface**:
   - Clear contract between data preparation and visualization
   - Enables multiple data preparation strategies
   - Consistent format regardless of time period or extraction method

5. **Maritime Units and Terminology**:
   - Consistently uses nautical miles (nm) not kilometers (km)
   - Uses standardized field names (lat, lng, GT)
   - Properly handles AIS transmission locations (tloc)

## Environment Setup

The project requires:
- Python 3.8+
- MySQL database with traffic data
- Required packages listed in requirements.txt

Configuration is managed through environment variables in the .env file:
- DB_HOST: Database hostname
- DB_USER: Database username
- DB_PASSWORD: Database password
- DB_NAME: Database name

## Core Challenges and Solutions

1. **Challenge**: Processing large volumes of AIS data
   **Solution**: Parallel processing with asyncio and careful database query optimization

2. **Challenge**: Identifying meaningful maritime routes from noisy position data
   **Solution**: DBSCAN clustering with tuned parameters (eps=0.004) and preprocessing to remove noise

3. **Challenge**: Creating a decoupled, flexible architecture
   **Solution**: Standardized CSV interface between components with clear contracts

4. **Challenge**: Handling duplicate position reports
   **Solution**: Using IMO+tloc as a compound key for deduplication

5. **Challenge**: Accommodating different data preparation needs
   **Solution**: Multiple preparation strategies that all produce the same standardized output

6. **Challenge**: Database connection efficiency
   **Solution**: Direct cursor-based database queries instead of pandas read_sql

7. **Challenge**: Resource management with large datasets
   **Solution**: Limited concurrency with asyncio semaphores and batch processing

8. **Challenge**: Distinguishing closely parallel routes
   **Solution**: Fine-tuned DBSCAN parameters (eps=0.004, min_samples=2) and vessel type segmentation

9. **Challenge**: Visualizing both common and uncommon routes
   **Solution**: Handling individual unclustered routes alongside aggregated route patterns

## Helpful Tips for Future Work

1. Always maintain the decoupling between data preparation and visualization
2. Ensure any new data preparation methods produce CSV files in the standardized format
3. Use maritime units consistently (nautical miles, knots)
4. When processing vessel position data:
   - Sort by timestamp
   - Handle time gaps appropriately (often indicating different trips)
   - Calculate speeds based on distances and time differences
   - Create unique trip identifiers
5. Use the haversine formula for calculating distances between geographic coordinates
6. Remember that the visualization component can work with any CSV in the standardized format, regardless of how it was produced
7. Use the `--debug` flag when troubleshooting data extraction issues
8. Adjust the `--max-concurrency` parameter based on your system's resources
9. For better route visualization:
   - Reduce DBSCAN epsilon (eps) to 0.003-0.005 for more detailed route differentiation
   - Set min_samples to 2 to capture less common routes
   - Color-code routes by vessel type for better interpretability
   - Use the `--routes_only` parameter to remove distracting port markers
10. Process full-year data in batches to avoid memory issues and improve reliability
11. Use the exact fleet values from the `v_loc` view (`lngtankers`, `lpgtankers`, `containers`, `carcarriers`, `bulkcarriers`, `ferrys`, `cruises`, `oiltankers`) for consistent vessel type visualization
