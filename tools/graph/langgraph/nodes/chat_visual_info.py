from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from tools.graph.langgraph.message.widgets import Widget, WidgetBase

@dataclass
class ChatVisualInfo:
    widget: Optional[Widget] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatVisualInfo':
        show_widget = data.get("show_widget", False)
        widget_type = data.get("widget_type")
        widget_data = data.get("widget_data", {})

        if show_widget and widget_type:
            # The data dict passed to Widget.from_dict should contain widget_type and data
            widget_dict = {"widget_type": widget_type, "data": widget_data}
            widget_instance = Widget.from_dict(widget_dict)
            return cls(widget=widget_instance)
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        if self.widget:
            widget_dict = self.widget.to_dict()
            # Flatten widget_type and data for ChatVisualInfo's direct representation
            return {
                "show_widget": True,
                "widget_type": widget_dict.get("widget_type"),
                "widget_data": widget_dict.get("data", {})
            }
        return {"show_widget": False, "widget_type": None, "widget_data": {}}