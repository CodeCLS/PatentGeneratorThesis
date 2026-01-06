"""
Debug utilities for graph validator - opens browser windows with agent outputs.
"""
import webbrowser
import tempfile
import os
from typing import Any
import json


def open_debug_browser(content: str, title: str = "Debug Output") -> None:
    """
    Open a browser window with debug content.
    
    Args:
        content: Text content to display
        title: Title for the debug window
    """
    # Create a temporary HTML file
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: monospace;
            padding: 20px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        h1 {{
            color: #4ec9b0;
            border-bottom: 2px solid #4ec9b0;
            padding-bottom: 10px;
        }}
        pre {{
            background-color: #252526;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #3e3e42;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <pre>{_escape_html(content)}</pre>
</body>
</html>"""
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        temp_file = f.name
    
    # Open in browser
    file_url = f"file://{temp_file}"
    webbrowser.open(file_url)
    
    # Note: We don't delete the temp file immediately so the browser can load it
    # It will be cleaned up when the OS cleans temp files


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;"))


def format_agent_output(agent_name: str, input_data: Any, output_data: Any, metadata: dict = None) -> str:
    """
    Format agent input/output for debug display.
    
    Args:
        agent_name: Name of the agent
        input_data: Input to the agent
        output_data: Output from the agent
        metadata: Optional metadata about the call
        
    Returns:
        Formatted string for display
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"AGENT: {agent_name}")
    lines.append("=" * 80)
    lines.append("")
    
    if metadata:
        lines.append("METADATA:")
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    
    lines.append("INPUT:")
    lines.append("-" * 80)
    if isinstance(input_data, str):
        lines.append(input_data)
    elif isinstance(input_data, dict):
        lines.append(json.dumps(input_data, indent=2))
    else:
        lines.append(str(input_data))
    lines.append("")
    
    lines.append("OUTPUT:")
    lines.append("-" * 80)
    if isinstance(output_data, str):
        lines.append(output_data)
    elif isinstance(output_data, dict):
        lines.append(json.dumps(output_data, indent=2))
    elif isinstance(output_data, list):
        lines.append(json.dumps(output_data, indent=2))
    else:
        lines.append(str(output_data))
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)

