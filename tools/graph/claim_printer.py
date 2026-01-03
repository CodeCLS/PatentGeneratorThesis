"""
Helper functions for printing drafted patent claims in a readable format.
"""
from typing import List
from tools.graph.claim_drafting_agent import DraftedClaim


def print_claims(claims: List[DraftedClaim], show_type: bool = False, show_parent: bool = False) -> None:
    """
    Print patent claims in a clean, numbered format.
    
    Args:
        claims: List of DraftedClaim objects
        show_type: Whether to show claim type (independent/dependent)
        show_parent: Whether to show parent claim number for dependent claims
    """
    if not claims:
        print("No claims to display.")
        return
    
    print("\n" + "="*80)
    print(f"PATENT CLAIMS ({len(claims)} total)")
    print("="*80)
    
    for claim in claims:
        # Build claim header
        header_parts = [f"{claim.claim_number}."]
        if show_type:
            header_parts.append(f"[{claim.type.upper()}]")
        if show_parent and claim.parent_claim_number:
            header_parts.append(f"(depends on claim {claim.parent_claim_number})")
        
        header = " ".join(header_parts)
        print(f"\n{header}")
        print(f"{claim.claim_text}")
    
    print("\n" + "="*80)


def print_claims_compact(claims: List[DraftedClaim]) -> None:
    """
    Print claims in a compact, patent-style format (just numbers and text).
    """
    if not claims:
        print("No claims to display.")
        return
    
    for claim in claims:
        print(f"{claim.claim_number}. {claim.claim_text}")


def print_claims_grouped(claims: List[DraftedClaim]) -> None:
    """
    Print claims grouped by type (independent first, then dependent).
    """
    if not claims:
        print("No claims to display.")
        return
    
    independent = [c for c in claims if c.type == "independent"]
    dependent = [c for c in claims if c.type == "dependent"]
    
    print("\n" + "="*80)
    print(f"INDEPENDENT CLAIMS ({len(independent)})")
    print("="*80)
    for claim in independent:
        print(f"\n{claim.claim_number}. {claim.claim_text}")
    
    if dependent:
        print("\n" + "="*80)
        print(f"DEPENDENT CLAIMS ({len(dependent)})")
        print("="*80)
        for claim in dependent:
            parent_info = f" (depends on claim {claim.parent_claim_number})" if claim.parent_claim_number else ""
            print(f"\n{claim.claim_number}. {claim.claim_text}{parent_info}")
    
    print("\n" + "="*80)

