# Graph Validator Chat - Next.js

This is the Next.js version of the Graph Validator Chat interface. All functionality including widgets and containers remains the same as the original version.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Make sure the Python backend server is running on port 8000:
```bash
# From the parent directory
cd ../..
python web_editor/graph_validator_chat/server.py
```

3. Run the Next.js development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Features

- ✅ All original functionality preserved
- ✅ All 11 widget types supported
- ✅ Triple search and filtering
- ✅ Real-time status updates
- ✅ Graph statistics display
- ✅ Recent changes tracking
- ✅ Notion-like UI design

## API Configuration

The Next.js app proxies API requests to the Python backend server running on `http://localhost:8000`. This is configured in `next.config.js` using Next.js rewrites.

If your Python server runs on a different port, update the `next.config.js` file accordingly.

## Production Build

To build for production:

```bash
npm run build
npm start
```

