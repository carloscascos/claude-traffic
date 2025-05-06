#!/bin/bash
# Maritime Route Analysis Pipeline
# This script runs the complete process to create a maritime highway map:
# 1. Data extraction from the traffic database
# 2. Route analysis and visualization

# Configuration
MONTHS=(9 10 11)  # September, October, November
YEAR=2024
DATA_DIR="data"
FINAL_CSV="${DATA_DIR}/cruise_routes_sept_oct_nov_2024.csv"
OUTPUT_MAP="${DATA_DIR}/cruise_highway_map_2024.html"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

echo "==============================================="
echo "Maritime Highway Map Generation Pipeline"
echo "==============================================="
echo "Parameters:"
echo "- Year: $YEAR"
echo "- Months: September, October, November"
echo "- Output CSV: $FINAL_CSV"
echo "- Output map: $OUTPUT_MAP"
echo "==============================================="

# Step 1: Extract data from the traffic database for each month
echo ""
echo "Step 1: Extracting vessel positions from database..."

# Initialize empty CSV to combine all data
> "$FINAL_CSV"

# Process each month
for MONTH in "${MONTHS[@]}"; do
    echo "  Processing month: $MONTH"
    MONTH_CSV="${DATA_DIR}/cruise_routes_${YEAR}_${MONTH}.csv"
    
    # Extract data for this month - limit to 10000 records for demonstration
    . env/bin/activate && python src/route_data_prep.py --year $YEAR --month $MONTH --limit 10000 --output "$MONTH_CSV"
    
    # Check if extraction was successful
    if [ ! -f "$MONTH_CSV" ]; then
        echo "Error: Failed to extract vessel positions for month $MONTH."
        continue
    fi
    
    # Append data to final CSV (skip header if not first file)
    if [ ! -s "$FINAL_CSV" ]; then
        cat "$MONTH_CSV" > "$FINAL_CSV"
    else
        tail -n +2 "$MONTH_CSV" >> "$FINAL_CSV"
    fi
    
    # Clean up temporary files
    # rm "$MONTH_CSV"
done

# Check if we have data
if [ ! -s "$FINAL_CSV" ]; then
    echo "Error: Failed to extract any vessel positions."
    exit 1
fi

# Step 2: Generate the maritime highway map
echo ""
echo "Step 2: Generating maritime highway map..."
. env/bin/activate && python src/cruise_highway_map.py --input "$FINAL_CSV" --output "$OUTPUT_MAP" --min_speed 0.2 --min_trip_points 5 --cluster_eps 0.02 --cluster_min_samples 2

# Check if map generation was successful
if [ ! -f "$OUTPUT_MAP" ]; then
    echo "Error: Failed to generate maritime highway map. $OUTPUT_MAP does not exist."
    exit 1
fi

echo ""
echo "==============================================="
echo "Pipeline completed successfully!"
echo "Maritime highway map saved to: $OUTPUT_MAP"
echo "Open this file in a web browser to view the map."
echo "==============================================="
