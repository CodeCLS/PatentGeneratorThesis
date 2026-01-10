"""
Widget dataclasses for different widget types.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union

from tools.graph.langgraph.info import TripleInfo


@dataclass
class Widget:
    """Base widget class."""
    widget_type: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary - override in subclasses."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Widget":
        """Create widget from dictionary based on widget_type."""
        widget_type = data.get("widget_type") or data.get("type", "")
        
        widget_classes = {
            "edges_widget": EdgesWidget,
            "graph_widget": GraphWidget,
            "graph_subsection_widget": GraphSubsectionWidget,
            "question_widget_general": QuestionWidgetGeneral,
            "question_widget_triple": QuestionWidgetTriple,
            "question_widget_entity": QuestionWidgetEntity,
            "question_widget_cluster_triple": QuestionWidgetClusterTriple,
            "validation_summary_widget": ValidationSummaryWidget,
            "patent_analysis_widget": PatentAnalysisWidget,
            "connection_check_widget": ConnectionCheckWidget,
            "suggestion_widget": SuggestionWidget,
        }
        
        widget_class = widget_classes.get(widget_type)
        if widget_class:
            return widget_class.from_dict(data)
        
        # Fallback to base widget
        return WidgetBase(widget_type=widget_type, data=data.get("data", {}))


@dataclass
class WidgetBase(Widget):
    """Generic widget for unknown types."""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": self.data,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WidgetBase":
        """Create from dictionary."""
        return cls(
            widget_type=data.get("widget_type") or data.get("type", ""),
            data=data.get("data", {}),
        )


@dataclass
class EdgesWidget(Widget):
    """Widget showing subject-relation-object edges as a list."""
    triples: List[Union[TripleInfo, dict]] = field(default_factory=list)
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "edges_widget"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        triples_data = [
            t.to_dict() if hasattr(t, 'to_dict') else t
            for t in self.triples
        ]
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {"triples": triples_data},
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EdgesWidget":
        """Create from dictionary."""
        from tools.graph.langgraph.info import TripleInfo
        
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            triples_data = widget_data.get("triples", [])
        else:
            triples_data = data.get("triples", [])
        
        triples = [
            TripleInfo.from_dict(t) if isinstance(t, dict) else t
            for t in triples_data
        ]
        
        return cls(triples=triples)


@dataclass
class GraphWidget(Widget):
    """Widget visualizing the full knowledge graph."""
    graph_data: Optional[Dict[str, Any]] = None
    entities: List[Dict[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "graph_widget"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "graph_data": self.graph_data,
                "entities": self.entities,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GraphWidget":
        """Create from dictionary."""
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            return cls(
                graph_data=widget_data.get("graph_data"),
                entities=widget_data.get("entities", []),
            )
        return cls()


@dataclass
class GraphSubsectionWidget(Widget):
    """Widget visualizing a focused part of the knowledge graph."""
    graph_data: Optional[Dict[str, Any]] = None
    entities: List[Dict[str, str]] = field(default_factory=list)
    focus_entity: Optional[str] = None
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "graph_subsection_widget"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "graph_data": self.graph_data,
                "entities": self.entities,
                "focus_entity": self.focus_entity,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GraphSubsectionWidget":
        """Create from dictionary."""
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            return cls(
                graph_data=widget_data.get("graph_data"),
                entities=widget_data.get("entities", []),
                focus_entity=widget_data.get("focus_entity"),
            )
        return cls()


@dataclass
class QuestionWidgetGeneral(Widget):
    """Widget asking an open-ended question with text input."""
    question: str = ""
    placeholder: Optional[str] = None
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "question_widget_general"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "question": self.question,
                "placeholder": self.placeholder,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QuestionWidgetGeneral":
        """Create from dictionary."""
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            return cls(
                question=widget_data.get("question", ""),
                placeholder=widget_data.get("placeholder"),
            )
        return cls(question=data.get("question", ""))


@dataclass
class QuestionWidgetTriple(Widget):
    """Widget asking to confirm or correct a specific triple."""
    triple: Union[TripleInfo, dict] = field(default_factory=dict)
    question: Optional[str] = None
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "question_widget_triple"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        triple_data = self.triple.to_dict() if hasattr(self.triple, 'to_dict') else self.triple
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "triple": triple_data,
                "question": self.question,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QuestionWidgetTriple":
        """Create from dictionary."""
        from tools.graph.langgraph.info import TripleInfo
        
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            triple_data = widget_data.get("triple", {})
            triple = TripleInfo.from_dict(triple_data) if isinstance(triple_data, dict) else triple_data
            return cls(
                triple=triple,
                question=widget_data.get("question"),
            )
        return cls(triple={})


@dataclass
class QuestionWidgetEntity(Widget):
    """Widget asking to validate or explain an entity."""
    entity_name: str = ""
    entity_id: Optional[str] = None
    question: Optional[str] = None
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "question_widget_entity"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "entity_name": self.entity_name,
                "entity_id": self.entity_id,
                "question": self.question,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QuestionWidgetEntity":
        """Create from dictionary."""
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            return cls(
                entity_name=widget_data.get("entity_name", ""),
                entity_id=widget_data.get("entity_id"),
                question=widget_data.get("question"),
            )
        return cls(entity_name=data.get("entity_name", ""))


@dataclass
class QuestionWidgetClusterTriple(Widget):
    """Widget asking to rate a triple's importance within a cluster."""
    triples: List[Union[TripleInfo, dict]] = field(default_factory=list)
    cluster_name: Optional[str] = None
    question: Optional[str] = None
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "question_widget_cluster_triple"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        triples_data = [
            t.to_dict() if hasattr(t, 'to_dict') else t
            for t in self.triples
        ]
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "triples": triples_data,
                "cluster_name": self.cluster_name,
                "question": self.question,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QuestionWidgetClusterTriple":
        """Create from dictionary."""
        from tools.graph.langgraph.info import TripleInfo
        
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            triples_data = widget_data.get("triples", [])
            triples = [
                TripleInfo.from_dict(t) if isinstance(t, dict) else t
                for t in triples_data
            ]
            return cls(
                triples=triples,
                cluster_name=widget_data.get("cluster_name"),
                question=widget_data.get("question"),
            )
        return cls()


@dataclass
class ValidationSummaryWidget(Widget):
    """Widget summarizing validation results and success rate."""
    total_questions: int = 0
    answered_questions: int = 0
    success_rate: Optional[float] = None
    summary: Optional[str] = None
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "validation_summary_widget"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "total_questions": self.total_questions,
                "answered_questions": self.answered_questions,
                "success_rate": self.success_rate,
                "summary": self.summary,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ValidationSummaryWidget":
        """Create from dictionary."""
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            return cls(
                total_questions=widget_data.get("total_questions", 0),
                answered_questions=widget_data.get("answered_questions", 0),
                success_rate=widget_data.get("success_rate"),
                summary=widget_data.get("summary"),
            )
        return cls()


@dataclass
class PatentAnalysisWidget(Widget):
    """Widget displaying patent status, risk, and metadata."""
    patent_id: Optional[str] = None
    status: Optional[str] = None
    risk_level: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "patent_analysis_widget"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "patent_id": self.patent_id,
                "status": self.status,
                "risk_level": self.risk_level,
                "metadata": self.metadata,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PatentAnalysisWidget":
        """Create from dictionary."""
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            return cls(
                patent_id=widget_data.get("patent_id"),
                status=widget_data.get("status"),
                risk_level=widget_data.get("risk_level"),
                metadata=widget_data.get("metadata", {}),
            )
        return cls()


@dataclass
class ConnectionCheckWidget(Widget):
    """Widget validating logical connections and flagging issues."""
    connections: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "connection_check_widget"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "connections": self.connections,
                "issues": self.issues,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConnectionCheckWidget":
        """Create from dictionary."""
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            return cls(
                connections=widget_data.get("connections", []),
                issues=widget_data.get("issues", []),
            )
        return cls()


@dataclass
class SuggestionWidget(Widget):
    """Widget presenting AI suggestions with accept or dismiss actions."""
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Set widget_type."""
        self.widget_type = "suggestion_widget"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "widget_type": self.widget_type,
            "type": self.widget_type,
            "data": {
                "suggestions": self.suggestions,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SuggestionWidget":
        """Create from dictionary."""
        widget_data = data.get("data", {})
        if isinstance(widget_data, dict):
            return cls(
                suggestions=widget_data.get("suggestions", []),
            )
        return cls()

