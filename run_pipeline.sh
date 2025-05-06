#!/bin/bash
# Maritime Route Analysis Pipeline
# This script runs the complete processing pipeline:
# 1. Data preparation (prep.py)
# 2. Parameter tuning (tune.py) 
# 3. Route rendering (render.py)
# 4. Display results (display.py)

# Default configuration
PORT="Huelva"
YEAR=2024
MONTH=01
WINDOW_DAYS=30
DATA_DIR="data"
RESULTS_DIR="results"
SERVER_PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --year)
      YEAR="$2"
      shift 2
      ;;
    --month)
      MONTH="$2"
      shift 2
      ;;
    --window)
      WINDOW_DAYS="$2"
      shift 2
      ;;
    --server-port)
      SERVER_PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--port PORT] [--year YEAR] [--month MONTH] [--window DAYS] [--server-port PORT]"
      exit 1
      ;;
  esac
done

# Generate filenames based on parameters
PORT_LC=${PORT,,}  # Convert to lowercase
OUTPUT_CSV="${DATA_DIR}/${PORT_LC}_${YEAR}_${MONTH}_w${WINDOW_DAYS}.csv"
PARAMS_JSON="${DATA_DIR}/${PORT_LC}_${YEAR}_${MONTH}_params.json"
OUTPUT_HTML="${RESULTS_DIR}/${PORT_LC}_${YEAR}_${MONTH}_map.html"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"

echo "==============================================="
echo "Maritime Route Analysis Pipeline"
echo "==============================================="
echo "Parameters:"
echo "- Port: $PORT"
echo "- Year: $YEAR"
echo "- Month: $MONTH"
echo "- Window days: $WINDOW_DAYS"
echo "- Output CSV: $OUTPUT_CSV"
echo "- Parameters: $PARAMS_JSON" 
echo "- Output map: $OUTPUT_HTML"
echo "- Server port: $SERVER_PORT"
echo "==============================================="

# Step 1: Data Preparation
echo ""
echo "Step 1: Preparing vessel position data..."
source ./env/bin/activate && python src/prep.py --port "$PORT" --year $YEAR --month $MONTH --window-days $WINDOW_DAYS --output "$OUTPUT_CSV"

# Check if preparation was successful
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "Error: Failed to prepare vessel position data."
    exit 1
fi

# Step 2: Parameter Tuning
echo ""
echo "Step 2: Tuning DBSCAN parameters..."
source ./env/bin/activate && python src/tune.py --input "$OUTPUT_CSV" --output "$PARAMS_JSON"

# Check if tuning was successful
if [ ! -f "$PARAMS_JSON" ]; then
    echo "Error: Failed to tune parameters."
    exit 1
fi

# Step 3: Rendering the map
echo ""
echo "Step 3: Rendering maritime highway map..."
source ./env/bin/activate && python src/render.py --params "$PARAMS_JSON" --output "$OUTPUT_HTML" --by_vessel_type

# Check if rendering was successful
if [ ! -f "$OUTPUT_HTML" ]; then
    echo "Error: Failed to render maritime highway map."
    exit 1
fi

# Step 4: Display the results
echo ""
echo "Step 4: Starting display server..."
echo "The server will start on http://localhost:$SERVER_PORT"
echo "Press Ctrl+C to stop the server when done."
source ./env/bin/activate && python src/display.py --port $SERVER_PORT --latest --open-browser

echo ""
echo "==============================================="
echo "Pipeline completed successfully!"
echo "==============================================="
