#!/bin/bash

# Monitor script for the rendering job
PID=32140
LOG_FILE="results/render_monitor.log"

echo "Starting to monitor rendering job with PID $PID" > $LOG_FILE
echo "Time: $(date)" >> $LOG_FILE
echo "----------------------------" >> $LOG_FILE

# Check if the process is still running
while ps -p $PID > /dev/null; do
    # Get CPU and memory usage
    CPU=$(ps -p $PID -o %cpu | tail -n 1)
    MEM=$(ps -p $PID -o %mem | tail -n 1)
    
    # Check if any output file has been created
    if [ -f "results/huelva_2024_window30_map.html" ]; then
        FILE_SIZE=$(du -h results/huelva_2024_window30_map.html | cut -f1)
        echo "$(date) - Process running (CPU: $CPU%, MEM: $MEM%) - Output file size: $FILE_SIZE" >> $LOG_FILE
    else
        echo "$(date) - Process running (CPU: $CPU%, MEM: $MEM%) - No output file yet" >> $LOG_FILE
    fi
    
    # Sleep for 15 seconds before checking again
    sleep 15
done

# Check if the process has completed normally
if [ ! -f "results/huelva_2024_window30_map.html" ]; then
    echo "$(date) - Process has terminated but no output file was created. Check for errors." >> $LOG_FILE
else
    FILE_SIZE=$(du -h results/huelva_2024_window30_map.html | cut -f1)
    echo "$(date) - Process has completed. Final HTML file size: $FILE_SIZE" >> $LOG_FILE
fi

echo "----------------------------" >> $LOG_FILE
echo "Monitoring complete" >> $LOG_FILE
