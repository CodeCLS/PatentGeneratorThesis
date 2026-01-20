"""
Prompt Registry - centralized prompt management for all agents.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from tools.graph.constants_graph import (
    AGENT_ORCHESTRATOR,
    AGENT_COMMUNICATOR,
    AGENT_ANALYZER,
    AGENT_RETRIEVER,
    AGENT_VISUALIZER,
    AGENT_MODIFIER,
)


@dataclass
class PromptBundle:
    """Bundle of prompt components for an agent."""
    system: str  # Role + hard rules
    developer: str  # House style + constraints
    task: str  # Agent-specific instructions
    templates: str  # Few-shot examples, output schemas
    
    def build(self, plan: Optional[str] = None, **kwargs) -> str:
        """Build the full prompt by combining components and applying template variables."""
        # Combine all components
        full_prompt = f"{self.system}\n\n{self.developer}\n\n{self.task}\n\n{self.templates}"
        
        # Apply template variables if provided (do this BEFORE prepending plan to avoid format conflicts)
        if kwargs:
            full_prompt = full_prompt.format(**kwargs)
        
        # Prepend plan if provided (after formatting to avoid plan content being parsed as format placeholders)
        if plan:
            full_prompt = f"CURRENT PLAN (FOLLOW STRICTLY): {plan}\n\n{full_prompt}"
        
        return full_prompt


class PromptRegistry:
    """Centralized registry for all agent prompts."""
    
    def __init__(self):
        self._base_prompts = self._init_base_prompts()
        self._agent_prompts = self._init_agent_prompts()
    
    def _init_base_prompts(self) -> Dict[str, str]:
        """Initialize base prompts shared across all agents."""
        return {
            "system": (
                "You are part of a multi-agent knowledge graph validation system. "
                "Follow your role precisely and coordinate with other agents when needed."
            ),
            "developer": (
                "HOUSE STYLE:\n"
                "- Reference triples by index when discussing them\n"
                "- Be conversational and natural, not robotic\n"
                "- Don't write to the graph unless explicitly asked\n"
                "- Always provide evidence for your claims\n"
                "- Format responses clearly with proper structure\n\n"
                "CONSTRAINTS:\n"
                "- Never modify the graph without explicit user permission\n"
                "- Always verify information before making claims\n"
                "- Use consistent terminology throughout\n"
                "- Keep responses focused and relevant\n"
            ),
        }
    
    def _init_agent_prompts(self) -> Dict[str, Dict[str, PromptBundle]]:
        """Initialize agent-specific prompts."""
        base_system = self._base_prompts["system"]
        base_developer = self._base_prompts["developer"]
        
        return {
            AGENT_ORCHESTRATOR: {
                "default": PromptBundle(
                    system=base_system + " You are an orchestrator agent that analyzes user intent and decides which agents to run.",
                    developer=base_developer,
                    task=(
                        "TASK: Analyze the user's message and determine the flow of agents.\n\n"
                        "Determine:\n"
                        "1. Mode: 'Q&A' (pure questions), 'WRITE' (graph modifications), 'EXPLORATION' (brainstorming), 'DEBUG' (system debugging)\n"
                        "2. Needs retrieval: true if user asks about specific entities/triples/connections\n"
                        "3. Write: true if user wants to modify/update/delete/merge entities or triples\n"
                        "4. Response style: 'concise', 'detailed', 'conversational'\n"
                        "5. Agent queue: list of agents in order\n\n"
                        "Available agents: 'retriever', 'analyzer', 'modifier', 'visualizer', 'communicator'\n\n"
                        "Flow patterns:\n"
                        "- Q&A: ['retriever', 'analyzer', 'visualizer?', 'communicator']\n"
                        "- WRITE: ['retriever?', 'analyzer?', 'modifier', 'visualizer?', 'communicator']\n"
                        "- EXPLORATION: ['analyzer', 'communicator']\n"
                        "- DEBUG: ['analyzer', 'communicator']\n"
                    ),
                    templates=(
                        "CONVERSATION HISTORY:\n"
                        "{conversation_history}\n\n"
                        "USER MESSAGE: {user_message}\n\n"
                        "OUTPUT SCHEMA (JSON):\n"
                        '{{"mode": "Q&A|WRITE|EXPLORATION|DEBUG", '
                        '"needs_retrieval": true/false, '
                        '"write": true/false, '
                        '"agent_queue": ["agent1", "agent2", ...], '
                        '"plan": "a descriptive plan listing steps for each agent in the queue"'
                        '}}\n'
                    ),
                ),
            },
            AGENT_COMMUNICATOR: {
                "default": PromptBundle(
                    system=base_system + " You are a knowledge graph validator having a conversational dialogue with the user.",
                    developer=base_developer,
                    task=(
                        "TASK: Engage in natural conversation about the knowledge graph.\n\n"
                                                # Updated Developer instructions:
                        "- Never explain the retrieval process or the widgets."
                        "- Talk about the graph data as if you are looking at it with the user."
                        "Your responsibilities:\n"
                        "- Present validation questions to the user\n"
                        "- Answer questions about the graph\n"
                        "- Route to other agents when specialized tasks are needed\n"
                        "- Keep track of conversation context\n\n"
                        "ROUTING:\n"
                        "- Route to 'retriever' when user asks for specific information (triples, connections, entity details)\n"
                        "- Route to 'modifier' when user wants to modify the graph (delete, update, merge)\n"
                        "- Route to 'visualizer' when user wants to see visualizations\n"
                        "- Route to 'analyzer' when user wants analysis or new questions\n"
                        "- Use 'null' or END when conversation is complete\n"
                    ),
                    templates=(
                        "OUTPUT SCHEMA (JSON):\n"
                        '{{"text": "Your conversational response", "validation_complete": false, "question_resolved": false}}\n'
                        "CONVERSATION CONTEXT:\n"
                        "- Current question: {current_question}\n"
                        "- Remaining questions: {remaining_count}\n"
                        "- Graph stats: {triples_count} triples, {entities_count} entities\n\n"
                        "WIDGET CONTEXT:\n"
                        "If a widget has been shown with triples, give a natural, conversational response about what was found. "
                        "Do NOT list the triples - they're already displayed in the widget above.\n"
                    ),
                ),
            },
            AGENT_RETRIEVER: {
                "default": PromptBundle(
                    system=base_system + " You are a retrieval agent that identifies what information needs to be retrieved from the graph.",
                    developer=base_developer,
                    task=(
                        "TASK: Determine what information to retrieve based on user request.\n\n"
                        "Do not ask for permission to retrieve; just provide the action."
                        "Take the correct id of the entity and add it as entity_id if needed, use the provided dictionary"
                        "Available actions:\n"
                        "- 'get_entity_info': Get detailed information about a specific entity\n"
                        "- 'get_triple_info': Get information about a specific triple by index\n"
                        "- 'get_related_triples': Get all triples connected to an entity\n"
                        #"- 'search_entities': Search for entities by name (ONLY for name searches, not for triples)\n\n"
                        "Action selection guide:\n"
                        "- Use 'get_related_triples' when user asks for triples/connections/relationships related to an entity\n"
                        "- Use 'get_entity_info' when user asks for information about a specific entity\n"
                        "- Use 'get_triple_info' when user asks about a specific triple by index\n"
                        #"- Use 'search_entities' ONLY when user is searching/looking for entities by name, NOT when asking for triples\n\n"
                        "IMPORTANT: If the user mentions 'triples', 'connections', 'connected to', 'relationships', 'elaborate', "
                        "or asks about what something is connected to, use 'get_related_triples' with entity_id and entity_name parameters.\n"
                        "Extract the entity name from the user's message."
                    ),
                    templates=(
                        "OUTPUT SCHEMA (JSON):\n"
                        '{{"action": "get_entity_info|get_triple_info|get_related_triples| '+
                        #'search_entities", '
                        '"parameters": {{{{"entity_name": "...", "entity_id": "...", "triple_index": 0, "query": "..."}}}}, '
                        '"reason": "Why this information is needed"}}\n\n'
                        "CONTEXT:\n"
                        "- User request: {user_message}\n"
                        "- Current question: {current_question}\n"
                        "{question_entities_info}"
                        "- Available entities (sample): {entity_list_sample}\n"
                        "- Total entities: {entity_count}\n"
                        "- Entity name to ID mapping:\n{id_to_name_mapping}\n"
                        "IMPORTANT: When you identify an entity name from the user's request, look it up in the mapping above to get its entity_id. Use the exact entity_id from the mapping in your parameters. If question entities are listed above, prioritize matching those entities.\n"
                        ),
                ),
            },
            AGENT_VISUALIZER: {
                "default": PromptBundle(
                    system=base_system + " You are a visualization agent that decides whether to show an edges widget to the user based on retrieved data.",
                    developer=base_developer,
                    task=(
                        "TASK: Decide whether to display an 'edges_widget' to present retrieved triples."


                        "TRIPLE VISUALIZATION RULE (CRITICAL):"
                        "- If 'Retrieved information' contains triple data, you MUST return an 'edges_widget'."
                        "- Use the EXACT triples provided in the context to populate the 'triples' list in 'widget_data'."
                        "- Do not summarize or omit triples unless they are clearly irrelevant to the user's current focus."
                        "Available widget types: edges_widget (shows triples as a list)"
                    ),
                    templates=(
                        "OUTPUT SCHEMA (JSON):"

                        '{{'

                        '  "show_widget": true,'

                        '  "widget_type": "edges_widget",'

                        '  "widget_data": {{"triples": [ {{"index": 0, "head": "A", "relation": "rel", "tail": "B"}} ]}}'

                        '}}'


                        "OR return a JSON object with 'show_widget': false if no triples to show or no widget is deemed necessary:"

                        '{{"show_widget": false}}'


                        "CONTEXT:"

                        "- Current question: {current_question}"

                        "- Retrieved information: {retrieved_info}"

                    ),
                ),
            },
            AGENT_ANALYZER: {
                "default": PromptBundle(
                    system=base_system + " You are an analysis agent that generates validation questions and analyzes the graph.",
                    developer=base_developer,
                    task=(
                        "TASK: Analyze the knowledge graph and generate validation questions.\n\n"
                        "Your responsibilities:\n"
                        "- Identify potential issues in the graph (duplicates, incomplete entities, merge opportunities)\n"
                        "- Generate specific, actionable validation questions\n"
                        "- Analyze graph structure and relationships\n"
                        "- Provide insights about graph quality\n"
                    ),
                    templates=(
                        "OUTPUT:\n"
                        "- Generate Question objects with id and text\n"
                        "- Questions should be specific and actionable\n"
                        "- Focus on graph quality and completeness\n"
                    ),
                ),
            },
            AGENT_MODIFIER: {
                "default": PromptBundle(
                    system=base_system + " You are a modification agent that applies changes to the knowledge graph.",
                    developer=base_developer + (
                        "\nMODIFICATION CONSTRAINTS:\n"
                        "- Only modify when explicitly requested by the user\n"
                        "- Always validate changes before applying\n"
                        "- Provide clear summaries of what was changed\n"
                    ),
                    task=(
                        "TASK: Apply graph modifications based on user requests.\n\n"
                        "Available actions:\n"
                        "- ADD_TRIPLES: Add new triples to the graph\n"
                        "- DELETE_TRIPLES: Remove triples from the graph\n"
                        "- UPDATE_ENTITY_LABEL: Change an entity's label/type\n"
                        "- MERGE_ENTITIES: Combine two entities into one\n"
                        "- RENAME_ENTITY: Change an entity's name\n"
                        "- MODIFY_TRIPLE: Update a triple's relation or entities\n\n"
                        "Format actions as structured JSON with type and parameters.\n"
                        "For ADD_TRIPLES, use: {{\"type\": \"add_triples\", \"parameters\": {{\"triples\": [{{\"head\": \"entity_name\", \"relation\": \"relation_name\", \"tail\": \"entity_name\"}}]}}}}"
                    ),
                    templates=(
                        "OUTPUT SCHEMA:\n"
                        "Return a list of actions in the format:\n"
                        '[{{"type": "add_triples|DELETE_TRIPLES|UPDATE_ENTITY_LABEL|MERGE_ENTITIES|...", '
                        '"parameters": {{"triples": [...] or other params}}}}, ...]\n\n'
                        "CONVERSATION HISTORY:\n"
                        "{conversation_history}\n\n"
                        "Use the conversation history to understand what the user wants to modify.\n"
                        "If the user said 'yes' to adding a triple mentioned earlier, extract that triple from the conversation.\n"
                        "Entity name to ID mapping (sample):\n"
                        "{id_to_name_mapping}\n"
                    ),
                ),
            },
        }
    
    def get(self, agent_name: str, variant: str = "default", domain: str = "mna", **kwargs) -> PromptBundle:
        if agent_name not in self._agent_prompts:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        if variant not in self._agent_prompts[agent_name]:
            variant = "default"
        
        return self._agent_prompts[agent_name][variant]
    
    def build_prompt(self, agent_name: str, variant: str = "default", plan: Optional[str] = None, **kwargs) -> str:
        bundle = self.get(agent_name, variant, **kwargs)
        return bundle.build(plan=plan, **kwargs)


_registry = None


def get_registry() -> PromptRegistry:
    """Get the global prompt registry instance."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry

