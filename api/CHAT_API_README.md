# Chat API Implementation

This document describes the new chat application API endpoints that have been added to the FastAPI backend.

## Overview

The chat API provides endpoints for:
- User management (authentication handled by Next.js, database operations by FastAPI)
- Chat and messaging with streaming support
- Knowledge graph triples
- Document management (chat artifacts)
- File uploads
- Voting on messages
- Chat history

## Authentication

All endpoints (except user creation/verification) require JWT authentication. The JWT tokens are issued by Next.js (NextAuth.js) and validated by FastAPI using the `AUTH_SECRET` environment variable.

### JWT Token Structure
```json
{
  "id": "user-uuid",
  "email": "user@example.com",
  "type": "regular" | "guest",
  "iat": 1234567890,
  "exp": 1234567890
}
```

### Token Extraction
Tokens are extracted from:
1. `Authorization: Bearer <token>` header (preferred)
2. `next-auth.session-token` cookie (fallback)

## API Endpoints

### Users

#### `POST /api/users`
Create a new user (called by Next.js during registration).
- **Auth**: Not required
- **Body**: `{ "email": "user@example.com", "password": "hashed-password" }`
- **Response**: `201 Created` with user object

#### `GET /api/users?email={email}`
Get user by email (for Next.js password verification).
- **Auth**: Not required
- **Response**: Array with user object (includes password hash for verification)

#### `GET /api/users`
Get all users (for team management).
- **Auth**: Required
- **Response**: Array of user objects (without passwords)

#### `GET /api/users/{userId}`
Get user by ID.
- **Auth**: Required
- **Response**: User object

#### `POST /api/users/ensure`
Ensure user exists (create if not, return if exists). Used for OAuth flows.
- **Auth**: Not required
- **Body**: `{ "email": "user@example.com", "password": null }`
- **Response**: User object

### Chat

#### `POST /api/chat`
Create or continue a chat with streaming response.
- **Auth**: Required
- **Body**: Chat creation data with message
- **Response**: Server-Sent Events (SSE) stream
- **Rate Limited**: Yes (based on user type)
- **Headers**: Supports geolocation headers (`x-vercel-ip-*`)

#### `DELETE /api/chat?id={chatId}`
Delete a chat and all associated data.
- **Auth**: Required
- **Authorization**: User must own the chat

#### `GET /api/chat/{chatId}/messages`
Get all messages for a chat.
- **Auth**: Required
- **Authorization**: User must own the chat
- **Response**: Array of messages ordered by createdAt

### Knowledge Graph

#### `GET /api/knowledge-graph`
Get all triples for the authenticated user.
- **Auth**: Required
- **Response**: Array of triples

#### `POST /api/knowledge-graph`
Create a new triple.
- **Auth**: Required
- **Body**: `{ "subject": "...", "predicate": "...", "object": "..." }`

#### `PATCH /api/knowledge-graph`
Update a triple.
- **Auth**: Required
- **Authorization**: User must own the triple
- **Body**: `{ "id": "...", "subject": "...", "predicate": "...", "object": "..." }`

#### `DELETE /api/knowledge-graph?id={tripleId}`
Delete a triple.
- **Auth**: Required
- **Authorization**: User must own the triple

### Documents (Chat Artifacts)

#### `GET /api/document?id={documentId}`
Get document by ID (returns all versions).
- **Auth**: Required
- **Authorization**: User must own the document

#### `GET /api/document/all`
Get all documents for the authenticated user.
- **Auth**: Required

#### `POST /api/document?id={documentId}`
Create or update a document (creates new version).
- **Auth**: Required
- **Body**: `{ "title": "...", "content": "...", "kind": "text|code|image|sheet" }`

#### `DELETE /api/document?id={documentId}&timestamp={timestamp}`
Delete document versions created after the specified timestamp.
- **Auth**: Required
- **Authorization**: User must own the document

### Files

#### `POST /api/files/upload`
Upload a file.
- **Auth**: Required
- **Request**: `multipart/form-data` with `file` field
- **Validation**: File size (max 10MB), MIME type
- **Response**: `{ "url": "...", "pathname": "...", "contentType": "..." }`

### Votes

#### `GET /api/vote?chatId={chatId}`
Get all votes for a chat.
- **Auth**: Required
- **Authorization**: User must own the chat

#### `PATCH /api/vote`
Vote on a message (up or down).
- **Auth**: Required
- **Authorization**: User must own the chat
- **Body**: `{ "chatId": "...", "messageId": "...", "type": "up|down" }`

### History

#### `GET /api/history`
Get paginated chat history.
- **Auth**: Required
- **Query Params**: `limit`, `starting_after`, `ending_before`
- **Response**: `{ "chats": [...], "hasMore": true|false }`

#### `DELETE /api/history`
Delete all chats for the authenticated user.
- **Auth**: Required
- **Response**: `{ "deletedCount": number }`

## Database Schema

All new tables are defined in `api/database/sql_models.py`:
- `User` - User accounts
- `Chat` - Chat conversations
- `Message_v2` - Chat messages
- `Vote_v2` - Message votes
- `Document` - Chat artifacts (versioned)
- `KnowledgeGraphTriple` - Knowledge graph triples
- `Stream` - Stream records
- `Suggestion` - Document suggestions (optional)

## Configuration

Required environment variables:
- `AUTH_SECRET` - **CRITICAL**: Same secret used by Next.js for JWT signing/verification
- `DATABASE_TYPE` - `local`, `postgres`, or `supabase`
- `POSTGRES_*` - PostgreSQL connection settings
- `RATE_LIMIT_MESSAGES_GUEST` - Rate limit for guest users (default: 50)
- `RATE_LIMIT_MESSAGES_REGULAR` - Rate limit for regular users (default: 1000)
- `MAX_FILE_SIZE_MB` - Maximum file upload size (default: 10)
- `STORAGE_TYPE` - File storage type: `local`, `s3`, `vercel` (default: `local`)

## Rate Limiting

Rate limiting is implemented for chat messages:
- **Guest users**: Limited messages per 24 hours (configurable)
- **Regular users**: Higher limits (configurable)
- **Window**: 24-hour rolling window
- **Error**: Returns `429 Too Many Requests` if exceeded

## File Uploads

Supported file types:
- **Images**: JPEG, PNG, GIF, WebP, SVG
- **Documents**: PDF, Word, Excel, PowerPoint
- **Text**: Plain text, CSV, Markdown, HTML
- **Other**: JSON, RTF, ZIP

File size limit: 10MB (configurable)

## Streaming Chat

The chat endpoint uses Server-Sent Events (SSE) for streaming responses:
- **Format**: `data-appendMessage: {...}\n\n`
- **Events**: `data-appendMessage`, `data-finishMessage`, `data-chat-title`
- **Headers**: Proper SSE headers set automatically

## Error Handling

All errors follow the format:
```json
{
  "error": "Error message",
  "code": "error_code:category"
}
```

Error categories:
- `bad_request:*` - Invalid request
- `unauthorized:*` - Not authenticated
- `forbidden:*` - Not authorized
- `not_found:*` - Resource not found
- `rate_limit:*` - Rate limit exceeded

## AI Service Integration

The `AIService` class in `api/services/ai_service.py` is a placeholder. You need to integrate it with your actual AI provider (OpenAI, Anthropic, etc.).

The service should:
- Accept messages and model selection
- Stream responses in SSE format
- Handle geolocation hints
- Generate chat titles

## Next Steps

1. **Integrate AI Provider**: Update `api/services/ai_service.py` with your AI provider
2. **Configure Storage**: Set up file storage (S3, Azure Blob, etc.) if not using local
3. **Set AUTH_SECRET**: Ensure `AUTH_SECRET` matches Next.js configuration
4. **Database Migration**: Run `python -m api.database.init_db` to create tables
5. **Test Endpoints**: Use the interactive docs at `/docs` to test endpoints

## Notes

- All timestamps are in UTC
- UUIDs are used for all IDs
- Cascade deletes are configured for related data
- Indexes are created for common query patterns
- The API supports both local and PostgreSQL/Supabase storage

