"""
Helper script to save and load notebook variables across kernel restarts.
"""

import pickle
import os
from typing import Any, Dict, Optional

STATE_FILE = 'notebook_state.pkl'


def save_state(**kwargs) -> None:
    """
    Save variables to disk for persistence across kernel restarts.
    
    Usage:
        save_state(G=G, triples=triples, id_to_name=id_to_name)
    """
    # Load existing state if it exists
    existing_state = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                existing_state = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing state: {e}")
    
    # Merge with new variables
    existing_state.update(kwargs)
    
    # Save merged state
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(existing_state, f)
    
    print(f"✓ Saved {len(kwargs)} variable(s) to {STATE_FILE}")
    print(f"  Variables: {', '.join(kwargs.keys())}")


def load_state(variable_names: Optional[list] = None) -> Dict[str, Any]:
    """
    Load variables from disk.
    
    Usage:
        # Load all variables
        state = load_state()
        G = state['G']
        triples = state['triples']
        
        # Or load specific variables
        state = load_state(['G', 'triples'])
    """
    if not os.path.exists(STATE_FILE):
        print(f"No saved state found at {STATE_FILE}")
        return {}
    
    try:
        with open(STATE_FILE, 'rb') as f:
            state = pickle.load(f)
        
        if variable_names:
            # Return only requested variables
            result = {name: state.get(name) for name in variable_names}
            print(f"✓ Loaded {len([v for v in result.values() if v is not None])} variable(s) from {STATE_FILE}")
            return result
        else:
            # Return all variables
            print(f"✓ Loaded {len(state)} variable(s) from {STATE_FILE}")
            return state
    except Exception as e:
        print(f"Error loading state: {e}")
        return {}


def clear_state() -> None:
    """Clear saved state file."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print(f"✓ Cleared saved state at {STATE_FILE}")
    else:
        print(f"No saved state found at {STATE_FILE}")


def list_saved_variables() -> list:
    """List all variable names in saved state."""
    if not os.path.exists(STATE_FILE):
        return []
    
    try:
        with open(STATE_FILE, 'rb') as f:
            state = pickle.load(f)
        return list(state.keys())
    except Exception as e:
        print(f"Error reading state: {e}")
        return []

