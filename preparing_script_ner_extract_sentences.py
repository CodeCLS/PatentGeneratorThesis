import epo_ops
import os
import spacy
import xml.etree.ElementTree as ET
import json
# Instantiate client.
client = epo_ops.Client(key=os.getenv("EPO_CONSUMER_KEY"), secret=os.getenv("EPO_CONSUMER_SECRET_KEY"))

print(os.getenv("EPO_CONSUMER_KEY"))
# Retrieve bibliography data.
response = client.published_data(

  # publication, application, priority
  reference_type="publication",

  # docdb, epodoc
  input=epo_ops.models.Docdb("1000000", "EP", "A1"),

  # optional, defaults to biblio in case of published_data
  endpoint="biblio",

  # optional, list of constituents
  constituents=[],
)

# Retrieve description.
response = client.published_data(
  reference_type="publication",
  input=epo_ops.models.Docdb("1000000", "EP", "A1"),
  endpoint="description",
)
print("Description " + response.text)

xml_text = response.text

# Parse XML
root = ET.fromstring(xml_text)

# EPO OPS uses the "epodoc" namespace for content
ns = {
    "ops": "http://www.epo.org/exchange",
    "ep": "http://www.epo.org/fulltext"
}

# Collect all description paragraphs
paragraphs = root.findall(".//ep:p", ns)

description_text = "\n".join([p.text.strip() for p in paragraphs if p.text])

print("---- DESCRIPTION ----")
print(description_text)

nlp = spacy.load("en_core_web_sm")

doc = nlp(description_text)
sentences = [sent.text.strip() for sent in doc.sents]
print("Sentences: " + str(sentences))

def get_last_id(filename):
    if not os.path.exists(filename):
        return 0  # no file yet → start at 1

    last_id = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj and isinstance(obj["id"], int):
                    last_id = max(last_id, obj["id"])
            except Exception:
                continue  # skip malformed lines

    return last_id
output_file = "sentences.txt"
current_id = get_last_id(output_file)

with open(output_file, "a", encoding="utf-8") as f:
    for sentence in sentences:
        current_id += 1
        entry = {
            "id": current_id,
            "data": {"text": sentence}
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Appended {len(sentences)} sentences to {output_file}.")
print(f"Last assigned ID: {current_id}")
