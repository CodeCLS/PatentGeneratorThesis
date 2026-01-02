# Example notebook cell for using the triple editor
# Copy this into a cell in your Main.ipynb

from web_editor import start_triple_editor, get_updated_triples

# Assuming you have triples from your pipeline
# For example, after FAISS merging, LLM filtering, etc.
# triples = filtered_triples  # or whatever your variable name is

# Start the editor (browser opens automatically)
start_triple_editor(triples, port=5000)

# ============================================
# NOW EDIT IN THE BROWSER:
# ============================================
# 1. Click on any triple in the left sidebar
# 2. Edit the head entity label, name, etc.
# 3. Edit the relation
# 4. Edit the tail entity label, name, etc.
# 5. Click "Save Changes"
# 6. Repeat for other triples
# 7. Use search/filter to find specific triples
# 8. When done, close the browser window
# ============================================

# After editing, get the updated triples
updated_triples = get_updated_triples()

# Now continue with your claim clustering
# updated_triples is ready to use with your ClaimClusterer

