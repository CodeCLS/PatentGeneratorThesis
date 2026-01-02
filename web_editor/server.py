"""
Helper module to start the triple editor web server from a Jupyter notebook.
"""
from typing import List, Optional
import threading
import webbrowser
import time
from tools.graph.Triple import Triple
from web_editor.app import app, initialize_repository, run_server


def start_triple_editor(
    triples: List[Triple],
    port: int = 5000,
    open_browser: bool = True,
    debug: bool = False
) -> None:
    """
    Start the triple editor web server and open it in a browser.
    
    This function initializes the repository with the provided triples,
    starts a Flask server, and optionally opens it in the default browser.
    The server runs in a separate thread, so the notebook remains responsive.
    
    Args:
        triples: List of Triple objects to edit
        port: Port number to run the server on (default: 5000)
        open_browser: Whether to automatically open the browser (default: True)
        debug: Enable Flask debug mode (default: False)
    
    Example:
        >>> from tools.graph.Triple import Triple
        >>> from web_editor.server import start_triple_editor
        >>> 
        >>> # Assuming you have a list of triples
        >>> start_triple_editor(my_triples, port=5000)
        >>> # Server is now running at http://localhost:5000
        >>> # Make your edits in the browser, then close the window
        >>> # The triples list is updated in real-time
    """
    if not triples:
        raise ValueError("Triples list cannot be empty")
    
    # Initialize repository
    repo = initialize_repository(triples)
    print(f"✓ Initialized repository with {len(triples)} triples")
    print(f"✓ Repository contains {repo.get_entity_count()} unique entities")
    
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=run_server,
        args=(port, debug, open_browser),
        daemon=True
    )
    server_thread.start()
    
    print(f"✓ Server starting on http://localhost:{port}")
    if open_browser:
        print("✓ Browser will open automatically...")
    print("\n" + "="*60)
    print("TRIPLE EDITOR INSTRUCTIONS:")
    print("="*60)
    print("1. Edit triples and entities in the web interface")
    print("2. Changes are saved in real-time to the repository")
    print("3. When done, close the browser window")
    print("4. The updated triples are available via get_updated_triples()")
    print("="*60 + "\n")


def get_updated_triples() -> Optional[List[Triple]]:
    """
    Get the updated list of triples from the repository.
    
    This should be called after editing in the web interface to retrieve
    the modified triples.
    
    Returns:
        List of Triple objects, or None if repository not initialized
    
    Example:
        >>> from web_editor.server import get_updated_triples
        >>> 
        >>> # After editing in the web interface
        >>> updated_triples = get_updated_triples()
        >>> # Now use updated_triples for your ClaimCluster algorithms
    """
    from web_editor.app import repo
    
    if repo is None:
        print("Warning: Repository not initialized. Call start_triple_editor() first.")
        return None
    
    triples = list(repo.get_all_triples().values())
    print(f"✓ Retrieved {len(triples)} triples from repository")
    return triples


def get_repository():
    """
    Get the repository instance (for advanced usage).
    
    Returns:
        EnhancedEntityTripleRepository instance, or None if not initialized
    """
    from web_editor.app import repo
    return repo

