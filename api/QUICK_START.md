# Quick Start Guide - Running the API

## Prerequisites

1. ✅ Dependencies installed (already done)
2. ⚠️ Database configured (if using PostgreSQL/Supabase)
3. ⚠️ Environment variables set (especially `AUTH_SECRET`)

## Running the API

### Method 1: Using the run script (Recommended)

```bash
python api/run.py
```

### Method 2: Using uvicorn directly

```bash
uvicorn api.main:app --reload
```

### Method 3: Using Python directly

```bash
python -m api.main
```

## Configuration

Before running, make sure your `.env` file has:

```env
# Required for chat API
AUTH_SECRET=your-secret-here  # MUST match Next.js AUTH_SECRET

# Database (if using PostgreSQL/Supabase)
DATABASE_TYPE=supabase  # or postgres, or local
POSTGRES_HOST=db.xxxxx.supabase.co
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
POSTGRES_DB=postgres

# Optional
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

## Initialize Database (if using PostgreSQL/Supabase)

Before first run, create the database tables:

```bash
python -m api.database.init_db
```

## Access the API

Once running, the API will be available at:

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Troubleshooting

### "Module not found" errors
- Make sure you're in the project root directory
- Verify dependencies are installed: `pip list | grep fastapi`

### "Database not initialized" errors
- If using PostgreSQL/Supabase, run: `python -m api.database.init_db`
- Check your database connection settings in `.env`

### "AUTH_SECRET not set" warnings
- Set `AUTH_SECRET` in your `.env` file
- Must match the secret used by your Next.js application

### Port already in use
- Change the port in `.env`: `API_PORT=8001`
- Or kill the process using port 8000

