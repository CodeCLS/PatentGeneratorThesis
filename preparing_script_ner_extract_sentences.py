import os
import json
import xml.etree.ElementTree as ET

import epo_ops
import spacy
from dotenv import load_dotenv

from kg.formatting.formatting_manager import FormattingManager

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

load_dotenv()

client = epo_ops.Client(
    key=os.getenv("EPO_CONSUMER_KEY"),
    secret=os.getenv("EPO_CONSUMER_SECRET_KEY")
)

print("Using EPO_CONSUMER_KEY:", os.getenv("EPO_CONSUMER_KEY"))

# -------------------------------------------------------------------
# Helper: Get last assigned ID from file
# -------------------------------------------------------------------

def get_last_id(filename: str) -> int:
    if not os.path.exists(filename):
        return 0  # no file yet → start at 0

    last_id = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines

            if isinstance(obj, dict) and "id" in obj and isinstance(obj["id"], int):
                last_id = max(last_id, obj["id"])

    return last_id

# -------------------------------------------------------------------
# 1. Retrieve patent description from EPO OPS
# -------------------------------------------------------------------

response = client.published_data(
    reference_type="publication",
    input=epo_ops.models.Docdb("1000000", "EP", "A1"),
    endpoint="description",
)
xml_text = response.text
print("Description XML received.")

# -------------------------------------------------------------------
# 2. Parse XML and extract description paragraphs
# -------------------------------------------------------------------

ns = {
    "ops": "http://www.epo.org/exchange",
    "ep": "http://www.epo.org/fulltext",
}

root = ET.fromstring(xml_text)
paragraphs = root.findall(".//ep:p", ns)

description_text = "\n".join(
    [p.text.strip() for p in paragraphs if p.text]
)

print("---- DESCRIPTION ----")
print(description_text)

# -------------------------------------------------------------------
# 3. Use spaCy to get initial sentence segmentation
# -------------------------------------------------------------------

nlp = spacy.load("en_core_web_trf")
doc = nlp(description_text)

base_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
print("Base sentences from spaCy:", len(base_sentences))

# -------------------------------------------------------------------
# 4. Use FormattingManager for further splitting
# -------------------------------------------------------------------

formatting_manager = FormattingManager()

final_sentences: list[str] = []
for base in base_sentences:
    # FormattingManager.split always returns list[str]
    sub_sentences = formatting_manager.split(base)
    print("sub: " + str(sub_sentences))
    final_sentences.extend(sub_sentences)

print("Final sentences after FormattingManager.split:", len(final_sentences))

# -------------------------------------------------------------------
# 5. Append sentences (with IDs) to output file
# -------------------------------------------------------------------

output_file = "sentences.txt"
current_id = get_last_id(output_file)

with open(output_file, "a", encoding="utf-8") as f:
    for sentence in final_sentences:
        current_id += 1
        entry = {
            "id": current_id,
            "data": {"text": sentence.text},
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Appended {len(final_sentences)} sentences to {output_file}.")
print(f"Last assigned ID: {current_id}")
