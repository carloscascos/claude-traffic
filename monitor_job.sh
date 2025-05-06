#!/bin/bash

# Monitor script for the running job
PID=18310
LOG_FILE="data/huelva_job_monitor.log"

echo "Starting to monitor job with PID $PID" > $LOG_FILE
echo "Time: $(date)" >> $LOG_FILE
echo "----------------------------" >> $LOG_FILE

# Check if the process is still running
while ps -p $PID > /dev/null; do
    # Get CPU and memory usage
    CPU=$(ps -p $PID -o %cpu | tail -n 1)
    MEM=$(ps -p $PID -o %mem | tail -n 1)
    
    # Check if any output files have been created
    if [ -f "data/huelva_2024_window30.csv" ]; then
        CSV_SIZE=$(du -h data/huelva_2024_window30.csv | cut -f1)
        echo "$(date) - Process running (CPU: $CPU%, MEM: $MEM%) - CSV size: $CSV_SIZE" >> $LOG_FILE
    else
        echo "$(date) - Process running (CPU: $CPU%, MEM: $MEM%) - No output file yet" >> $LOG_FILE
    fi
    
    # Sleep for 30 seconds before checking again
    sleep 30
done

# Check if the process has completed normally
if [ ! -f "data/huelva_2024_window30.csv" ]; then
    echo "$(date) - Process has terminated but no output file was created. Check for errors." >> $LOG_FILE
else
    CSV_SIZE=$(du -h data/huelva_2024_window30.csv | cut -f1)
    CSV_ROWS=$(wc -l data/huelva_2024_window30.csv | cut -d' ' -f1)
    echo "$(date) - Process has completed. Final CSV size: $CSV_SIZE, Rows: $CSV_ROWS" >> $LOG_FILE
fi

echo "----------------------------" >> $LOG_FILE
echo "Monitoring complete" >> $LOG_FILE
