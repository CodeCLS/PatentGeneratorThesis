"""
Fix Jinja2 escape import issue.
Import this BEFORE importing Flask or web_editor.

Usage:
    import fix_jinja2_import  # Must be first
    from flask import Flask  # Now this will work
"""

# Patch jinja2.escape before Flask tries to import it
import jinja2

if not hasattr(jinja2, 'escape'):
    try:
        from markupsafe import escape
        jinja2.escape = escape
        print("✓ Patched jinja2.escape from markupsafe")
    except ImportError:
        # Fallback implementation
        def escape(s):
            if s is None:
                return ''
            s = str(s)
            return (s.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;'))
        jinja2.escape = escape
        print("✓ Patched jinja2.escape with fallback")

