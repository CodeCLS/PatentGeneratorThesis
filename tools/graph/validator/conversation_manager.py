"""
Conversation Manager: Manages conversation history.
"""
from typing import List, Dict, Optional
from tools.graph.validator_types import ConversationTurn


class ConversationManager:
    """Manages conversation history."""
    
    def __init__(self):
        # Per-question conversation history
        self.conversation_history: Dict[str, List[ConversationTurn]] = {}
        # Global chat history: [{"role": "user/bot", "content": "..."}]
        self.global_conversation_history: List[Dict[str, str]] = []
    
    def add_turn(self, question_id: str, user_answer: str, bot_response: str, turn_number: int) -> None:
        """Add a conversation turn for a specific question."""
        if question_id not in self.conversation_history:
            self.conversation_history[question_id] = []
        
        self.conversation_history[question_id].append(
            ConversationTurn(
                user_answer=user_answer,
                bot_response=bot_response,
                turn_number=turn_number,
            )
        )
    
    def get_turns(self, question_id: str) -> List[ConversationTurn]:
        """Get all conversation turns for a specific question."""
        return self.conversation_history.get(question_id, [])
    
    def add_global_message(self, role: str, content: str) -> None:
        """Add a message to the global conversation history."""
        self.global_conversation_history.append({"role": role, "content": content})
    
    def get_global_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get global conversation history, optionally limited to last N messages."""
        if limit:
            return self.global_conversation_history[-limit:]
        return self.global_conversation_history.copy()
    
    def format_conversation_history(self, question_id: str, max_turns: int = 5) -> str:
        """Format conversation history for a question as text."""
        turns = self.get_turns(question_id)
        if not turns:
            return ""
        
        text = "\n\nCONVERSATION HISTORY FOR THIS QUESTION:\n"
        text += "=" * 60 + "\n"
        for turn in turns[-max_turns:]:  # Show last N turns
            text += f"\nTurn {turn.turn_number}:\n"
            text += f"  User: {turn.user_answer}\n"
            text += f"  Bot: {turn.bot_response[:200]}{'...' if len(turn.bot_response) > 200 else ''}\n"
        text += "=" * 60 + "\n"
        return text
    
    def format_global_history(self, max_messages: int = 10) -> str:
        """Format global conversation history as text."""
        if not self.global_conversation_history:
            return ""
        
        text = "\n\nFULL CONVERSATION HISTORY:\n"
        text += "=" * 60 + "\n"
        for i, msg in enumerate(self.global_conversation_history[-max_messages:], 1):
            role = msg["role"].upper()
            content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
            text += f"\n{i}. {role}: {content}\n"
        text += "=" * 60 + "\n"
        return text
    
    def clear_question_history(self, question_id: str) -> None:
        """Clear conversation history for a specific question."""
        self.conversation_history.pop(question_id, None)
    
    def clear_all(self) -> None:
        """Clear all conversation history."""
        self.conversation_history = {}
        self.global_conversation_history = []

