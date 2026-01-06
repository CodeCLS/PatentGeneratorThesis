import epo_ops
from dotenv import load_dotenv
import os
import spacy
import xml.etree.ElementTree as ET
import json
load_dotenv()
class PatentProvider:

    def __init__(self):


        # Instantiate client.
        self.client = epo_ops.Client(key=os.getenv("EPO_CONSUMER_KEY"), secret=os.getenv("EPO_CONSUMER_SECRET_KEY"))
    def getDescription(self,patentId):
            response = self.client.published_data(
            reference_type="publication",
            input=epo_ops.models.Docdb(str(patentId), "EP", "A1"),
            endpoint="description",
            )

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

            return description_text
