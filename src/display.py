#!/usr/bin/env python3
"""
Maritime Routes Display Server

This script starts a simple HTTP server to display generated maritime route maps.
It automatically detects all HTML files in the results directory and provides
a clean interface to browse and view them.

Usage:
    python display.py [--port 8000] [--latest] [--open-browser]
"""

import os
import sys
import argparse
import http.server
import socketserver
import webbrowser
from datetime import datetime
import threading
import time
import mimetypes
from urllib.parse import parse_qs, urlparse
import html

# Ensure we can find our results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

def find_html_files(directory=RESULTS_DIR):
    """Find all HTML files in the specified directory with their modification times."""
    html_files = []
    
    if not os.path.exists(directory):
        print(f"Error: Results directory {directory} does not exist.")
        return []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.html'):
            full_path = os.path.join(directory, filename)
            mod_time = os.path.getmtime(full_path)
            # Convert to datetime for display
            mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            html_files.append({
                'filename': filename,
                'path': full_path,
                'mtime': mod_time,
                'mtime_str': mod_time_str
            })
    
    # Sort by modification time (newest first)
    html_files.sort(key=lambda x: x['mtime'], reverse=True)
    return html_files

def get_latest_html_file(directory=RESULTS_DIR):
    """Find the most recently modified HTML file in the specified directory."""
    html_files = find_html_files(directory)
    if html_files:
        return html_files[0]
    return None

def generate_file_listing(html_files, current_url="/"):
    """Generate an HTML page listing all available visualization files."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maritime Routes Map Viewer</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1e68a7;
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .file-list {{
            list-style: none;
            padding: 0;
        }}
        .file-list li {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        .file-list li:hover {{
            background-color: #f5f7fa;
        }}
        .file-list li a {{
            color: #1e68a7;
            text-decoration: none;
            display: block;
        }}
        .file-list li a:hover {{
            text-decoration: underline;
        }}
        .file-date {{
            color: #777;
            font-size: 0.9em;
            margin-left: 10px;
        }}
        .latest-badge {{
            background-color: #28a745;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        .back-button {{
            display: inline-block;
            margin-bottom: 20px;
            color: #1e68a7;
            text-decoration: none;
        }}
        .back-button:hover {{
            text-decoration: underline;
        }}
        .refresh-button {{
            background-color: #1e68a7;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            float: right;
            text-decoration: none;
        }}
        .refresh-button:hover {{
            background-color: #17517f;
        }}
        .button-container {{
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .view-latest {{
            background-color: #28a745;
            padding: 8px 16px;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
            text-decoration: none;
            display: inline-block;
        }}
        .view-latest:hover {{
            background-color: #218838;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Maritime Routes Map Viewer</h1>
        
        <div class="button-container">
"""
    
    # Add the "View Latest" button if there are any HTML files
    if html_files:
        latest_file = html_files[0]['filename']
        html_content += f'            <a href="/view/{latest_file}" class="view-latest">View Latest Map</a>\n'
    
    # Add refresh button
    html_content += f"""            <a href="{current_url}" class="refresh-button">Refresh List</a>
        </div>
        
        <h2>Available Visualizations</h2>
"""
    
    # Generate the file listing
    if html_files:
        html_content += '        <ul class="file-list">\n'
        
        for i, file_info in enumerate(html_files):
            filename = file_info['filename']
            mod_time = file_info['mtime_str']
            
            # Add a "Latest" badge for the most recent file
            latest_badge = '<span class="latest-badge">Latest</span>' if i == 0 else ''
            
            html_content += f'            <li><a href="/view/{html.escape(filename)}">{html.escape(filename)}{latest_badge}</a><span class="file-date">{mod_time}</span></li>\n'
        
        html_content += '        </ul>\n'
    else:
        html_content += '        <p>No visualization files found. Generate a map first using the pipeline.</p>\n'
    
    html_content += """    </div>
</body>
</html>
"""
    return html_content

class RoutesMapHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for serving maritime routes maps."""
    
    def __init__(self, *args, **kwargs):
        self.results_dir = RESULTS_DIR
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        
        # Extract path and query parameters
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        
        # Handle the root path
        if path == "/" or path == "":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Find all HTML files
            html_files = find_html_files(self.results_dir)
            
            # Generate and send the listing page
            listing_html = generate_file_listing(html_files, self.path)
            self.wfile.write(listing_html.encode())
            return
        
        # Handle requests to view a specific file
        elif path.startswith("/view/"):
            filename = path[6:]  # Remove the "/view/" prefix
            file_path = os.path.join(self.results_dir, filename)
            
            if os.path.exists(file_path) and os.path.isfile(file_path) and filename.lower().endswith('.html'):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                
                # Read and send the file
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                # File not found or not an HTML file
                self.send_error(404, "File not found")
            return
        
        # Handle static files (for future enhancements)
        elif path.startswith("/static/"):
            # Remove "/static/" prefix
            file_rel_path = path[8:]
            file_path = os.path.join(os.path.dirname(self.results_dir), 'static', file_rel_path)
            
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                
                # Determine content type
                content_type, _ = mimetypes.guess_type(file_path)
                if content_type:
                    self.send_header("Content-type", content_type)
                else:
                    self.send_header("Content-type", "application/octet-stream")
                    
                self.end_headers()
                
                # Read and send the file
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                # File not found
                self.send_error(404, "File not found")
            return
            
        # All other paths
        else:
            self.send_error(404, "File not found")
            return

def open_browser(url):
    """Open the browser after a short delay to ensure the server is running."""
    time.sleep(1)
    webbrowser.open(url)

def main():
    """Main function to start the HTTP server."""
    parser = argparse.ArgumentParser(description='Start an HTTP server to display maritime route maps.')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on (default: 8000)')
    parser.add_argument('--latest', action='store_true', help='Open the latest map directly instead of the file listing')
    parser.add_argument('--open-browser', action='store_true', help='Automatically open the browser')
    
    args = parser.parse_args()
    
    # Create the HTTP server
    handler = RoutesMapHandler
    httpd = socketserver.TCPServer(("", args.port), handler)
    
    print(f"Server started at http://localhost:{args.port}")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Find the latest HTML file
    latest_file = get_latest_html_file(RESULTS_DIR)
    if latest_file:
        print(f"Latest map: {latest_file['filename']} (modified {latest_file['mtime_str']})")
    else:
        print("No maps found in the results directory.")
    
    # Open the browser if requested
    if args.open_browser:
        url = f"http://localhost:{args.port}"
        
        # If --latest is specified and there's a latest file, open it directly
        if args.latest and latest_file:
            url += f"/view/{latest_file['filename']}"
        
        # Start a thread to open the browser after a short delay
        threading.Thread(target=open_browser, args=(url,)).start()
    
    # Run the server
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()
