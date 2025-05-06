# Claude Traffic - Maritime Traffic Analysis Tool

## Overview

Claude Traffic is a Python-based toolkit for analyzing and visualizing maritime vessel traffic data. The project provides tools to process vessel position data, analyze movement patterns, and generate interactive maps showing maritime routes with filtering by vessel type (fleet).

## Estructura del proyecto / Project Structure

- `src/`: Source code directory
  - `prep.py`, `tune.py`, `render.py`, `display.py`: Core pipeline modules
  - Additional modules for different analysis approaches
- `data/`: Directory for input/output data files
- `docs/`: Additional project documentation
- `CLAUDE.md`: Detailed project documentation

## Requisitos / Requirements

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
  - python-dotenv
  - seaborn
  - requests
  - fastapi
  - uvicorn
  - pytest
  - black
  - flake8

## Configuración / Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Run the complete pipeline on the sample data:
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

## Licencia / License

This project is under the MIT License.
