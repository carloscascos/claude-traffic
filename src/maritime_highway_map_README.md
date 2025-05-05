# Maritime Highway Map Generator

This tool creates interactive maps showing maritime routes with line widths proportional to vessel traffic gross tonnage. It visualizes common shipping lanes and cruise vessel traffic patterns.

## Overview

The maritime highway map generator consists of:

1. **Data Extraction**: Fetches cruise vessel positions from the traffic database for a specified time period.

2. **Route Analysis**: Processes the position data to identify common routes traveled by multiple vessels.

3. **Visualization**: Creates an interactive HTML map where the width of each route is proportional to the total gross tonnage of vessels that traveled along it.

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - geopandas
  - numpy
  - folium
  - scikit-learn
  - matplotlib
  - shapely

## Installation

Install the required packages:

```bash
pip install pandas geopandas numpy folium scikit-learn matplotlib shapely
```

## Usage

### Quick Start

Run the complete pipeline with the default settings:

```bash
chmod +x run-maritime-analysis.sh
./run-maritime-analysis.sh
```

This will:
1. Extract cruise vessel positions from October 2024
2. Process the data to identify common routes
3. Generate an interactive HTML map at `../data/cruise_highway_map_oct2024.html`

### Step-by-Step Execution

#### 1. Extract Vessel Positions

```bash
python route_data_prep.py --year 2024 --month 10 --output "../data/cruise_routes_oct2024.csv"
```

Optional arguments:
- `--limit`: Maximum number of records to fetch (for testing)

#### 2. Generate Maritime Highway Map

```bash
python cruise_highway_map.py --input "../data/cruise_routes_oct2024.csv" --output "../data/cruise_highway_map_oct2024.html"
```

Optional arguments:
- `--min_speed`: Minimum speed (knots) for route points (default: 3.0)
- `--min_trip_points`: Minimum points for a valid trip (default: 10)
- `--cluster_eps`: DBSCAN eps parameter for route clustering (default: 0.01)
- `--cluster_min_samples`: DBSCAN min_samples parameter for route clustering (default: 2)

## How It Works

### Data Extraction

The `route_data_prep.py` script:

1. Connects to the traffic database using the `database.py` module
2. Queries the `coldiron_loc` table for cruise vessel positions in the specified month/year
3. Joins with the `imo` table to get gross tonnage (GT) information
4. Calculates time differences, distances, and speeds between consecutive points
5. Groups positions into discrete trips based on time gaps
6. Outputs a CSV file with processed position data

### Route Identification and Visualization

The `cruise_highway_map.py` script:

1. Loads the position data and filters out stationary points and short trips
2. Creates line geometries for each vessel trip
3. Uses DBSCAN clustering to identify common routes
4. Aggregates similar routes and calculates total gross tonnage
5. Creates an interactive map where line width is proportional to gross tonnage
6. Automatically identifies port locations based on stopped vessels

## Output

The generated map includes:
- Maritime routes with width proportional to total gross tonnage
- Popup information showing route details (vessel count, trips, tonnage)
- Automatically detected port locations
- Interactive controls for map navigation
- Legend explaining the visualization

## Customization

- Modify the DBSCAN parameters (`--cluster_eps` and `--cluster_min_samples`) to adjust how routes are clustered
- Change the minimum speed threshold (`--min_speed`) to filter out stationary positions
- Adjust the minimum points per trip (`--min_trip_points`) to filter out short or incomplete trips

## Integration with claude-traffic

This tool is designed to work with the claude-traffic project's existing database structure. It leverages the `database.py` module for database connections and uses the same data directory structure for outputs.

To add this functionality to the main claude-traffic CLI, add:

```python
# Command: maritime-highway
highway_parser = subparsers.add_parser('maritime-highway', 
                                      help='Generate maritime highway map')
highway_parser.add_argument('--year', type=int, default=2024, help='Year to analyze')
highway_parser.add_argument('--month', type=int, default=10, help='Month to analyze')
highway_parser.add_argument('--output', type=str, default=None, 
                           help='Output HTML file path (default: ../data/cruise_highway_map_YEARMONTH.html)')
```

And in the command processing section:

```python
elif args.command == 'maritime-highway':
    output_path = args.output
    if output_path is None:
        output_path = f"../data/cruise_highway_map_{args.year}{args.month:02d}.html"
    
    # Run the pipeline
    # First, extract data
    temp_csv = f"../data/temp_cruise_routes_{args.year}{args.month:02d}.csv"
    subprocess.run(['python', 'route_data_prep.py', 
                   '--year', str(args.year), 
                   '--month', str(args.month), 
                   '--output', temp_csv])
    
    # Then, generate the map
    subprocess.run(['python', 'cruise_highway_map.py',
                   '--input', temp_csv,
                   '--output', output_path])
    
    print(f"Maritime highway map saved to {output_path}")
```
