# LLM Patent Claim Generator API

REST API for processing documents and generating knowledge graphs from patent text.

## Features

- **Document Management**: Add, list, update, and delete documents
- **Pipeline Processing**: Run the full NLP pipeline (splitting, NER, coref, triple generation)
- **Triple Management**: Create, read, update, and delete knowledge graph triples
- **Graph Operations**: Visualize graphs and perform clustering
- **Database Support**: Local in-memory storage or PostgreSQL/Supabase

## Installation

1. Install dependencies:
```bash
pip install -r api/requirements.txt
```

2. Install spaCy model:
```bash
python -m spacy download en_core_web_trf
```

## Running the API

### Development
```bash
cd api
python main.py
```

Or using uvicorn directly:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### Documents

- `POST /documents` - Create a new document
- `GET /documents` - List documents (with pagination and filters)
- `GET /documents/{document_id}` - Get a document by ID
- `PATCH /documents/{document_id}` - Update a document
- `DELETE /documents/{document_id}` - Delete a document

### Pipeline

- `POST /pipeline/start` - Start processing a document
- `GET /pipeline/status/{job_id}` - Get pipeline job status
- `GET /pipeline/document/{document_id}/status` - Get latest status for a document

### Triples

- `POST /triples` - Create a new triple
- `POST /triples/batch` - Create multiple triples
- `GET /triples` - List triples (with filters)
- `GET /triples/{triple_id}` - Get a triple by ID
- `PATCH /triples/{triple_id}` - Update a triple
- `DELETE /triples/{triple_id}` - Delete a triple
- `DELETE /triples/document/{document_id}` - Delete all triples for a document

### Graph

- `POST /graph/visualize/{document_id}` - Generate graph visualization
- `POST /graph/cluster/{document_id}` - Cluster the graph

## Example Usage

### 1. Create a Document

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Patent Document",
    "text": "The present invention relates to a display device...",
    "source": "patent_12345"
  }'
```

### 2. Start Pipeline Processing

```bash
curl -X POST "http://localhost:8000/pipeline/start" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "your-document-id"
  }'
```

### 3. Check Pipeline Status

```bash
curl "http://localhost:8000/pipeline/status/your-job-id"
```

### 4. List Triples

```bash
curl "http://localhost:8000/triples?document_id=your-document-id"
```

### 5. Update a Triple

```bash
curl -X PATCH "http://localhost:8000/triples/your-triple-id" \
  -H "Content-Type: application/json" \
  -d '{
    "relation": "improved relation text"
  }'
```

### 6. Visualize Graph

```bash
curl -X POST "http://localhost:8000/graph/visualize/your-document-id?merge_relations=true"
```

## Configuration

Create a `.env` file in the project root:

```env
# Database type: local, postgres, or supabase
DATABASE_TYPE=local

# PostgreSQL settings (if using postgres)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=patent_kg

# Supabase settings (if using supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key

# API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
```

## Database Options

### Local (Default)
Uses in-memory storage. Data is lost on server restart.

### PostgreSQL/Supabase
To use PostgreSQL or Supabase, set `DATABASE_TYPE=postgres` or `DATABASE_TYPE=supabase` in your `.env` file and configure the connection settings.

Note: PostgreSQL/Supabase repository implementations need to be added (currently only local storage is implemented).

## Architecture

```
api/
├── main.py              # FastAPI application
├── config.py            # Configuration settings
├── database/
│   ├── models.py        # Data models
│   └── repository.py    # Repository pattern (local/PostgreSQL/Supabase)
├── schemas/
│   ├── documents.py     # Document Pydantic schemas
│   ├── triples.py       # Triple Pydantic schemas
│   └── pipeline.py      # Pipeline Pydantic schemas
├── services/
│   └── pipeline_service.py  # Pipeline execution logic
└── routers/
    ├── documents.py     # Document endpoints
    ├── triples.py       # Triple endpoints
    ├── pipeline.py      # Pipeline endpoints
    └── graph.py         # Graph endpoints
```

## Development

To add new endpoints:

1. Create schemas in `api/schemas/`
2. Add service methods in `api/services/`
3. Create router endpoints in `api/routers/`
4. Include router in `api/main.py`

## License

Same as the main project.




