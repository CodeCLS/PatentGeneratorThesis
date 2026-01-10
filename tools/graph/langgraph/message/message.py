"""
Message dataclass for chat messages.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Any
from enum import Enum

from tools.graph.langgraph.message.widgets import Widget


class MessageRole(str, Enum):
    """Message role types."""
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"


@dataclass
class Message:
    """Chat message with optional widget."""
    role: Union[MessageRole, str]
    content: str
    data: Optional[Any] = None
    widget: Optional[Widget] = None
    
    def __post_init__(self):
        """Convert string role to MessageRole enum if needed."""
        if isinstance(self.role, str):
            try:
                self.role = MessageRole(self.role)
            except ValueError:
                # Keep as string if not a valid enum value
                pass
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
        }
        if self.widget:
            result["widget"] = self.widget.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create from dictionary."""
        from tools.graph.langgraph.message.widgets import Widget
        
        widget = None
        if "widget" in data:
            widget_data = data["widget"]
            if isinstance(widget_data, dict):
                widget = Widget.from_dict(widget_data)
            elif isinstance(widget_data, Widget):
                widget = widget_data
        
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            widget=widget,
        )

