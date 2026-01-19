from langfuse import Langfuse

langfuse = Langfuse()

class LangfuseSummary():
    def __init__(self):
        pass
    def check_status():
        return langfuse.auth_check()
    def get_prompt():
        return langfuse.get_prompt()
    @staticmethod
    def from_chat_agent_state(state: dict):
        return {
            "messages": state.get("messages", []),
            "current_question": state.get("current_question", None),
            "questions": state.get("questions", []),
            "graph_nodes_count": state.get("graph_nodes_count", 0),
            "graph_edges_count": state.get("graph_edges_count", 0),
            "triples_count": state.get("triples_count", 0),
            "entities_count": state.get("entities_count", 0),
        }
    def create_dataset():
        dataset = langfuse.create_dataset(name="chat_agent_state", description="Chat agent state")
    def add_item_to_dataset(dataset: str, item: dict):
        langfuse.add_item_to_dataset(dataset=dataset, item=item)
