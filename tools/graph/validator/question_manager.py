"""
Question Manager: Manages questions (get, filter, etc.).
"""
from typing import List, Optional
from tools.graph.validator_types import Question


class QuestionManager:
    """Manages questions (get, filter, etc.)."""
    
    def __init__(self, questions: List[Question] = None):
        self.questions: List[Question] = questions or []
    
    def get_first_question(self) -> Optional[Question]:
        """
        Get the first (highest priority) unanswered question.
        
        Returns:
            The first unanswered Question object, or None if no questions
        """
        if not self.questions:
            return None
        
        # Filter out answered questions
        unanswered = [q for q in self.questions if not q.answered]
        if not unanswered:
            return None
        
        # Return highest priority unanswered question
        return unanswered[0]
    
    def get_all_questions(self) -> List[Question]:
        """
        Get all questions.
        
        Returns:
            List of all Question objects (including answered ones)
        """
        return self.questions.copy()
    
    def get_unanswered_questions(self) -> List[Question]:
        """
        Get all unanswered questions.
        
        Returns:
            List of unanswered Question objects
        """
        return [q for q in self.questions if not q.answered]
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get a question by its ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None
    
    def add_question(self, question: Question) -> None:
        """Add a question to the manager."""
        self.questions.append(question)
    
    def add_questions(self, questions: List[Question]) -> None:
        """Add multiple questions to the manager."""
        self.questions.extend(questions)
    
    def mark_question_answered(self, question_id: str) -> None:
        """Mark a question as answered."""
        question = self.get_question_by_id(question_id)
        if question:
            question.answered = True
    
    def clear(self) -> None:
        """Clear all questions."""
        self.questions = []

