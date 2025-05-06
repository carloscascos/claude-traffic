#!/usr/bin/env python3
"""
Results Display Module

This module formats analysis results for presentation, handles interactive
map functionality, and controls UI elements for filtering and selection.
"""

import os
import argparse
import webbrowser
import sys
import json
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socket

def find_free_port():
    """
    Find a free port to use for the HTTP server.
    
    Returns:
        int: Available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def create_server_thread(directory, port):
    """
    Create a simple HTTP server for displaying visualizations.
    
    Args:
        directory (str): Directory to serve
        port (int): Port to use
        
    Returns:
        threading.Thread: Server thread
    """
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
    
    server = HTTPServer(('localhost', port), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    return thread, server

def validate_input(input_path):
    """
    Validate that the input path exists and is a valid visualization file.
    
    Args:
        input_path (str): Path to check
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        return False
    
    # Check if it's an HTML file
    if input_path.endswith('.html'):
        return True
    
    # Check if it's a directory containing HTML files
    if os.path.isdir(input_path):
        html_files = [f for f in os.listdir(input_path) if f.endswith('.html')]
        if html_files:
            return True
        else:
            print(f"Error: Directory '{input_path}' does not contain any HTML files.")
            return False
    
    print(f"Error: Input '{input_path}' is not a valid HTML file or directory.")
    return False

def generate_index_html(input_path, output_path=None):
    """
    Generate an index.html file for a directory of visualizations.
    
    Args:
        input_path (str): Directory containing visualizations
        output_path (str, optional): Path to write the index.html file. Defaults to input_path/index.html.
        
    Returns:
        str: Path to the generated index.html file
    """
    if not os.path.isdir(input_path):
        print(f"Error: '{input_path}' is not a directory.")
        return None
    
    if output_path is None:
        output_path = os.path.join(input_path, 'index.html')
    
    # Find all HTML files in the directory
    html_files = [f for f in os.listdir(input_path) if f.endswith('.html') and f != 'index.html']
    html_files.sort()
    
    if not html_files:
        print(f"Error: No HTML files found in '{input_path}'.")
        return None
    
    # Generate the index.html content
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maritime Traffic Analysis Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1000px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .visualization-list {
            list-style-type: none;
            padding: 0;
        }
        .visualization-item {
            margin-bottom: 15px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .visualization-title {
            font-weight: bold;
            color: #333;
            font-size: 18px;
            margin-bottom: 5px;
        }
        .visualization-description {
            color: #666;
            margin-bottom: 10px;
        }
        .visualization-link {
            display: inline-block;
            background-color: #4CAF50;
            color: white;
            padding: 8px 15px;
            text-decoration: none;
            border-radius: 4px;
        }
        .visualization-link:hover {
            background-color: #45a049;
        }
        .timestamp {
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>Maritime Traffic Analysis Results</h1>
    <p>The following visualizations are available:</p>
    <ul class="visualization-list">
"""
    
    # Add each visualization file
    for i, html_file in enumerate(html_files):
        file_path = os.path.join(input_path, html_file)
        mod_time = os.path.getmtime(file_path)
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
        
        # Extract a reasonable title from the filename
        title = html_file.replace('_', ' ').replace('-', ' ').replace('.html', '')
        title = ' '.join(word.capitalize() for word in title.split())
        
        # Generate a description based on the filename
        if 'route' in html_file.lower():
            description = "Visualization of maritime routes and traffic patterns."
        elif 'port' in html_file.lower():
            description = "Analysis of port activity and vessel movements."
        elif 'vessel' in html_file.lower():
            description = "Information about vessel types and characteristics."
        else:
            description = "Maritime traffic visualization."
        
        html_content += f"""
        <li class="visualization-item">
            <div class="visualization-title">{title}</div>
            <div class="visualization-description">{description}</div>
            <a href="{html_file}" class="visualization-link">View Visualization</a>
            <div class="timestamp">Generated: {formatted_time}</div>
        </li>
"""
    
    # Close the HTML
    html_content += """
    </ul>
</body>
</html>
"""
    
    # Write the index.html file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated index.html at {output_path}")
    return output_path

def display_visualization(input_path, port=None):
    """
    Display a visualization by starting a web server and opening a browser.
    
    Args:
        input_path (str): Path to the visualization file or directory
        port (int, optional): Port to use for the HTTP server. If None, find a free port.
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Find a free port if not specified
    if port is None:
        port = find_free_port()
    
    # Determine what to display and in which directory
    if os.path.isdir(input_path):
        # It's a directory, serve the entire directory
        serve_dir = input_path
        
        # Check if an index.html exists, if not, generate one
        index_path = os.path.join(input_path, 'index.html')
        if not os.path.exists(index_path):
            index_path = generate_index_html(input_path)
        
        if index_path is None:
            return False
        
        # Relative path to open in the browser
        rel_path = 'index.html'
    else:
        # It's a file, serve from its directory
        serve_dir = os.path.dirname(input_path)
        if not serve_dir:
            serve_dir = '.'
        
        # Relative path to open in the browser
        rel_path = os.path.basename(input_path)
    
    # Create and start the server
    server_thread, server = create_server_thread(serve_dir, port)
    server_thread.start()
    
    # Compose the URL
    url = f"http://localhost:{port}/{rel_path}"
    
    print(f"Starting HTTP server on port {port}...")
    print(f"Opening {url} in your browser...")
    
    try:
        # Wait a moment for the server to start
        time.sleep(0.5)
        
        # Open the browser
        webbrowser.open(url)
        
        # Keep the server running until interrupted
        print("Server is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
        server_thread.join()
        print("Server stopped.")
    except Exception as e:
        print(f"Error: {e}")
        server.shutdown()
        server_thread.join()
        return False
    
    return True

def apply_interactive_filters(input_html, output_html=None, filters=None):
    """
    Apply additional interactive filtering to an existing HTML visualization.
    
    Args:
        input_html (str): Path to the input HTML file
        output_html (str, optional): Path to write the modified HTML. If None, overwrite input.
        filters (dict, optional): Additional filters to apply. Defaults to None.
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(input_html):
        print(f"Error: Input file '{input_html}' does not exist.")
        return False
    
    if output_html is None:
        output_html = input_html
    
    if filters is None:
        filters = {}
    
    try:
        # Read the input HTML
        with open(input_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check if this looks like a folium map
        if 'var map_' not in html_content:
            print(f"Warning: Input file '{input_html}' does not appear to be a folium map.")
        
        # Find the right spot to inject the filter controls 
        # (after the folium layer control, before closing body tag)
        body_close_pos = html_content.rfind('</body>')
        if body_close_pos == -1:
            print(f"Warning: Could not find </body> tag in '{input_html}'.")
            insert_pos = len(html_content)
        else:
            insert_pos = body_close_pos
        
        # Generate filter controls HTML
        filter_html = """
<div id="additional-filters" style="position: fixed; top: 60px; right: 10px; 
                                   background: white; padding: 10px; 
                                   border-radius: 5px; border: 2px solid #ccc; 
                                   z-index: 1000; max-width: 250px;">
    <div style="font-weight: bold; margin-bottom: 8px;">Additional Filters</div>
    <div id="filter-controls">
        <!-- Weight filter -->
        <div style="margin-bottom: 8px;">
            <label for="weight-filter" style="display: block; margin-bottom: 3px;">
                Minimum Gross Tonnage:
            </label>
            <input type="range" id="weight-filter" min="0" max="100" value="0" 
                   style="width: 100%;" 
                   oninput="document.getElementById('weight-value').innerText = this.value + '%'">
            <div style="text-align: right; font-size: 12px;">
                <span id="weight-value">0%</span>
            </div>
        </div>
        
        <!-- Date range filter (if timestamps available) -->
        <div style="margin-bottom: 8px;">
            <label style="display: block; margin-bottom: 3px;">
                Date Range (if available):
            </label>
            <select id="date-filter" style="width: 100%;">
                <option value="all">All Dates</option>
                <option value="last_week">Last Week</option>
                <option value="last_month">Last Month</option>
                <option value="last_quarter">Last 3 Months</option>
            </select>
        </div>
        
        <!-- Apply button -->
        <button onclick="applyFilters()" style="width: 100%; 
                                               padding: 5px; 
                                               background: #4CAF50; 
                                               color: white; 
                                               border: none; 
                                               border-radius: 3px; 
                                               cursor: pointer;">
            Apply Filters
        </button>
        
        <!-- Reset button -->
        <button onclick="resetFilters()" style="width: 100%; 
                                              margin-top: 5px;
                                              padding: 5px; 
                                              background: #f44336; 
                                              color: white; 
                                              border: none; 
                                              border-radius: 3px; 
                                              cursor: pointer;">
            Reset Filters
        </button>
    </div>
</div>

<script>
    // Store original visibility state
    var originalVisibility = {};
    var allPolylines = [];
    
    // Find all polylines when the page loads
    document.addEventListener('DOMContentLoaded', function() {
        // Find all SVG paths that represent polylines
        setTimeout(function() {
            let paths = document.querySelectorAll('path');
            paths.forEach(function(path, index) {
                // Check if this is likely a route line (polyline)
                if (path.getAttribute('stroke') && 
                    path.getAttribute('stroke-opacity') && 
                    !path.getAttribute('fill')) {
                    
                    allPolylines.push({
                        element: path,
                        weight: parseFloat(path.getAttribute('stroke-width')) || 1,
                        id: 'polyline-' + index
                    });
                    
                    // Store original visibility
                    originalVisibility['polyline-' + index] = true;
                }
            });
            
            // Set max value for weight filter based on actual weights
            let maxWeight = Math.max(...allPolylines.map(p => p.weight));
            let minWeight = Math.min(...allPolylines.map(p => p.weight));
            document.getElementById('weight-filter').max = maxWeight;
            document.getElementById('weight-filter').min = minWeight;
            document.getElementById('weight-filter').value = minWeight;
            document.getElementById('weight-value').innerText = minWeight + ' GT';
        }, 1000); // Wait for the map to fully load
    });
    
    function applyFilters() {
        // Get weight filter value
        let minWeight = parseFloat(document.getElementById('weight-filter').value);
        
        // Apply filters to each polyline
        allPolylines.forEach(function(polyline) {
            let visible = polyline.weight >= minWeight;
            
            // Apply visibility
            polyline.element.style.display = visible ? '' : 'none';
        });
    }
    
    function resetFilters() {
        // Reset the weight filter control
        let minWeight = Math.min(...allPolylines.map(p => p.weight));
        document.getElementById('weight-filter').value = minWeight;
        document.getElementById('weight-value').innerText = minWeight + ' GT';
        
        // Reset the date filter
        document.getElementById('date-filter').value = 'all';
        
        // Reset all polylines to original visibility
        allPolylines.forEach(function(polyline) {
            polyline.element.style.display = '';
        });
    }
</script>
"""
        
        # Insert the filter controls
        modified_html = html_content[:insert_pos] + filter_html + html_content[insert_pos:]
        
        # Write the modified HTML
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(modified_html)
        
        print(f"Applied interactive filters to {output_html}")
        return True
    
    except Exception as e:
        print(f"Error applying filters: {e}")
        return False

def main():
    """
    Main function to display maritime traffic visualizations.
    """
    parser = argparse.ArgumentParser(description='Display maritime traffic visualizations.')
    parser.add_argument('--input', type=str, required=True, 
                      help='Input HTML file or directory with visualizations')
    parser.add_argument('--port', type=int, help='Port to use for the HTTP server')
    parser.add_argument('--add-filters', action='store_true', 
                      help='Add interactive filters to HTML visualization')
    
    args = parser.parse_args()
    
    # Validate input
    if not validate_input(args.input):
        return
    
    # Apply filters if requested
    if args.add_filters and args.input.endswith('.html'):
        apply_interactive_filters(args.input)
    
    # Display the visualization
    display_visualization(args.input, args.port)

if __name__ == "__main__":
    main()
