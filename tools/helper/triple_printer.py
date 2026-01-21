"""
Helper functions for printing triples in a readable format.
"""
from typing import List
from tools.graph.data.Triple import Triple


def print_triples_vertical(triples: List[Triple], max_triples: int = None):
    """
    Print triples in a vertical, readable format.
    
    Args:
        triples: List of Triple objects to print
        max_triples: Maximum number of triples to print (None = print all)
    """
    if not triples:
        print("No triples to display.")
        return
    
    total = len(triples)
    to_print = triples[:max_triples] if max_triples else triples
    
    print(f"\n{'='*80}")
    print(f"TRIPLES ({len(to_print)}/{total})")
    print(f"{'='*80}\n")
    
    for i, triple in enumerate(to_print, 1):
        print(f"[{i}] {triple.head.name} ({triple.head.label})")
        print(f"    --[{triple.relation}]-->")
        print(f"    {triple.tail.name} ({triple.tail.label})")
        print()
    
    if max_triples and len(triples) > max_triples:
        print(f"... and {len(triples) - max_triples} more triples\n")


def print_triples_compact(triples: List[Triple], max_triples: int = None):
    """
    Print triples in a compact vertical format.
    
    Args:
        triples: List of Triple objects to print
        max_triples: Maximum number of triples to print (None = print all)
    """
    if not triples:
        print("No triples to display.")
        return
    
    total = len(triples)
    to_print = triples[:max_triples] if max_triples else triples
    
    print(f"\nTriples ({len(to_print)}/{total}):\n")
    
    for i, triple in enumerate(to_print, 1):
        print(f"{i}. {triple.head.name} --[{triple.relation}]--> {triple.tail.name}")
        print(f"   H: {triple.head.label} | T: {triple.tail.label}")
        print()
    
    if max_triples and len(triples) > max_triples:
        print(f"... and {len(triples) - max_triples} more triples\n")

