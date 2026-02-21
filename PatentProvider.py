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

    def getAbstract(self, patentId):
        """Fetch abstract for an EPO patent ID. Returns plain text or empty string on error."""
        try:
            response = self.client.published_data(
                reference_type="publication",
                input=epo_ops.models.Docdb(str(patentId), "EP", "A1"),
                endpoint="abstract",
            )
            xml_text = response.text
            root = ET.fromstring(xml_text)
            ns = {
                "ops": "http://www.epo.org/exchange",
                "ep": "http://www.epo.org/fulltext",
            }
            paragraphs = root.findall(".//ep:p", ns)
            abstract_text = " ".join([p.text.strip() for p in paragraphs if p.text])
            return abstract_text.strip() if abstract_text else ""
        except Exception:
            return ""

    def getClaims(self, patentId):
        """Fetch claims for an EPO patent ID. Returns list of plain text strings."""
        try:
            # EPO OPS: Published data, claims endpoint
            response = self.client.published_data(
                reference_type="publication",
                input=epo_ops.models.Docdb(str(patentId), "EP", "A1"),
                endpoint="claims",
            )
            xml_text = response.text
            root = ET.fromstring(xml_text)
            ns = {
                "ops": "http://www.epo.org/exchange",
                "ep": "http://www.epo.org/fulltext",
            }
            
            # Find all claim-text elements within claim elements
            claims_elements = root.findall(".//ep:claim", ns)
            claims = []
            for claim in claims_elements:
                texts = []
                # Often claim text is split across multiple tags or within <claim-text>
                for ct in claim.findall(".//ep:claim-text", ns):
                    # Recursively get all text content
                    text_parts = [t for t in ct.itertext() if t.strip()]
                    if text_parts:
                        texts.append(" ".join(text_parts))
                
                if texts:
                    claims.append(" ".join(texts))
            
            return claims
        except Exception as e:
            print(f"Error fetching claims for {patentId}: {e}")
            return []
