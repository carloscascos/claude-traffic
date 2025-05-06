# Claude Traffic - Maritime Traffic Analysis Tool

## Overview

Claude Traffic is a Python-based toolkit for analyzing and visualizing maritime vessel traffic data. The project provides tools to process vessel position data, analyze movement patterns, and generate interactive maps showing maritime routes with filtering by vessel type (fleet).

## Project Structure

- `src/`: Source code directory
  - `prep.py`: Data preparation module
  - `tune.py`: Parameter tuning module 
  - `render.py`: Visualization rendering module
  - `display.py`: Results display module
- `data/`: Directory for input/output data files
- `CLAUDE.md`: Detailed project documentation

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - folium
  - geopandas
  - matplotlib
  - scikit-learn
  - shapely
  - branca

## Quick Start

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy folium geopandas matplotlib scikit-learn shapely branca
   ```
4. Run the complete pipeline on the sample data:
   ```bash
   # Process the data
   python src/prep.py --input data/sample_vessel_data.csv --output data/processed_data.csv
   
   # Tune the parameters
   python src/tune.py --input data/processed_data.csv --output data/tuned_params.json
   
   # Generate the visualization
   python src/render.py --data data/processed_data.csv --params data/tuned_params.json --output data/maritime_routes_map.html
   
   # View the results
   python src/display.py --input data/maritime_routes_map.html --add-filters
   ```

## Key Features

- Processes vessel position data to identify routes and patterns
- Clusters similar routes using DBSCAN algorithm
- Generates interactive map visualizations with line widths proportional to vessel tonnage
- Provides filtering by vessel type (fleet) categories
- Adds interactive UI filters for data exploration

## Fleet Filtering

The pipeline properly processes the "fleet" column in the vessel data. Fleet information (vessel types such as ferries, container ships, bulk carriers, etc.) is presented as filterable options in the final HTML map visualization under the layer selector.

See the [CLAUDE.md](CLAUDE.md) file for detailed documentation.
