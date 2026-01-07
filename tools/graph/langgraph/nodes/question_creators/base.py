"""
Base class for question creators.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
from tools.graph.langgraph.question import Question

if TYPE_CHECKING:
    from tools.graph.langgraph.validator import GraphValidatorLangGraph


class BaseQuestionCreator(ABC):
    """Base class for all question creators."""
    
    def __init__(self, validator: "GraphValidatorLangGraph"):
        self.validator = validator
        self.api_repo = validator.api_repo
        self.triples = validator.triples
        self.graph = validator.graph
        self.id_to_name = validator.id_to_name
    
    @abstractmethod
    def generate_questions(self) -> List[Question]:
        """Generate questions for this creator's specific validation concern."""
        pass

