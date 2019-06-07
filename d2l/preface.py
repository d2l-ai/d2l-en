"""Functions for the preface chapter"""
import pygments
import inspect
import IPython

def show_source(obj):
    """Show the source codes """
    code = inspect.getsource(obj)
    formatter = pygments.formatters.HtmlFormatter()
    return IPython.display.HTML('<style type="text/css">{}</style>{}'.format(
        formatter.get_style_defs('.highlight'),
        pygments.highlight(code, pygments.lexers.PythonLexer(), formatter)))
