# Supabase PostgreSQL Setup Guide

This guide explains how to connect your FastAPI application to a Supabase PostgreSQL database.

## Prerequisites

1. A Supabase account and project (sign up at https://supabase.com)
2. Python 3.8+ with pip

## Step 1: Get Your Supabase Connection Details

1. Go to your Supabase project dashboard
2. Navigate to **Settings** → **Database**
3. Find the **Connection string** section
4. Copy the connection details:
   - **Host**: Your project's database host (e.g., `db.xxxxx.supabase.co`)
   - **Port**: Usually `5432` for direct connection, or `6543` for connection pooler
   - **Database**: Usually `postgres`
   - **User**: Usually `postgres`
   - **Password**: Your database password (found in Settings → Database → Database password)

## Step 2: Install Dependencies

Make sure you have the required packages installed:

```bash
pip install -r api/requirements.txt
```

## Step 3: Configure Environment Variables

Create a `.env` file in your project root (or set environment variables):

```env
# Database type: "local", "postgres", or "supabase"
DATABASE_TYPE=supabase

# PostgreSQL/Supabase connection settings
POSTGRES_HOST=db.xxxxx.supabase.co
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_database_password
POSTGRES_DB=postgres

# Optional: Supabase-specific settings (if using Supabase client)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

### Connection Pooler (Recommended for Production)

For better performance and connection management, use Supabase's connection pooler:

```env
POSTGRES_PORT=6543
```

The connection pooler is on port `6543` instead of `5432`.

## Step 4: Initialize Database Tables

Run the database initialization script to create all required tables:

```bash
python -m api.database.init_db
```

This will create the following tables:
- `documents`
- `sentences`
- `triples`
- `processing_jobs`

## Step 5: Start the API

Start your FastAPI application:

```bash
python api/run.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload
```

## Step 6: Verify Connection

1. Check the API health endpoint: `http://localhost:8000/health`
2. Check the API docs: `http://localhost:8000/docs`
3. Try creating a document via the API

## Troubleshooting

### Connection Errors

- **"Connection refused"**: Check that `POSTGRES_HOST` and `POSTGRES_PORT` are correct
- **"Authentication failed"**: Verify your `POSTGRES_PASSWORD` is correct
- **"Database does not exist"**: Make sure `POSTGRES_DB` is set to `postgres` (default Supabase database)

### SSL Connection Issues

If you encounter SSL errors, you may need to modify the connection string in `api/database/connection.py` to include SSL parameters:

```python
return (
    f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    f"?ssl=require"
)
```

### Table Creation Errors

If tables already exist, you can drop them first (use with caution!):

```bash
python -m api.database.init_db drop
python -m api.database.init_db
```

## Using Direct PostgreSQL (Non-Supabase)

If you're using a regular PostgreSQL database instead of Supabase, the setup is the same:

```env
DATABASE_TYPE=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database
```

## Switching Back to Local Storage

To use in-memory local storage instead:

```env
DATABASE_TYPE=local
```

No database connection is needed in this mode.



