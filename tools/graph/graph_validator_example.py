"""
Example usage of GraphValidator for interactive graph validation.
"""
from tools.graph.graph_validator import GraphValidator, Question, Response, ActionType
from tools.graph.Triple import Triple
from tools.graph.visualizer import GraphVisualizer
import networkx as nx


def example_usage():
    """Example of how to use GraphValidator."""
    
    # Example 1: Validate a graph
    visualizer = GraphVisualizer()
    # Assuming you have triples
    # triples = [...]  # Your list of Triple objects
    # G = visualizer.build_graph(triples)
    
    # Create validator
    validator = GraphValidator()
    
    # Analyze graph and/or triples
    # validator.analyze(graph=G, triples=triples, id_to_name=id_to_name)
    
    # Get first question
    first_question = validator.getFirstQuestion()
    if first_question:
        print(f"Question: {first_question.text}")
        print(f"Category: {first_question.category}")
        print(f"Priority: {first_question.priority}")
        print(f"Suggested actions: {len(first_question.suggested_actions)}")
        
        # Answer the question
        user_answer = input("Your answer: ")
        response = validator.answerQuestion(first_question.id, user_answer)
        
        print(f"\nResponse: {response.text}")
        print(f"Actions to perform: {len(response.actions)}")
        
        # Process actions
        for action in response.actions:
            print(f"  - {action.type.value}: {action.parameters}")
            if action.type == ActionType.SHOW_TRIPLES:
                triple_indices = action.parameters.get("triple_indices", [])
                print(f"    Show triples: {triple_indices}")
            elif action.type == ActionType.ASK_IMPORTANCE:
                triple_index = action.parameters.get("triple_index")
                print(f"    Ask importance of triple: {triple_index}")
            elif action.type == ActionType.OPEN_WIDGET:
                widget_type = action.parameters.get("widget_type")
                print(f"    Open widget: {widget_type}")
    
    # Get all questions
    all_questions = validator.getAllQuestions()
    print(f"\nTotal questions: {len(all_questions)}")
    
    # Process each question
    for question in all_questions:
        print(f"\n{question.id}: {question.text}")
        print(f"  Category: {question.category}, Priority: {question.priority}")
        
        # Show suggested actions
        for action in question.suggested_actions:
            print(f"  Suggested: {action.type.value}")


if __name__ == "__main__":
    example_usage()

