"""
Jinja2 compatibility shim for Flask.
This must be imported BEFORE Flask to fix the escape import issue.
"""
import sys

# Patch jinja2 to have escape if it doesn't
try:
    from jinja2 import escape
except ImportError:
    # Jinja2 3.1+ moved escape to markupsafe
    try:
        from markupsafe import escape
        # Inject it into jinja2 module for compatibility
        import jinja2
        jinja2.escape = escape
    except ImportError:
        # Fallback
        def escape(s):
            if s is None:
                return ''
            return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
        import jinja2
        jinja2.escape = escape

