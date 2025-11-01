import os
from typing import Optional


def run_notebook(input_path: str = "main.ipynb", output_path: str = "outputs/main_executed.ipynb", timeout: int = 1200, kernel_name: Optional[str] = None) -> str:
    """Execute a Jupyter notebook in-place and save the executed copy.

    Returns the output notebook path. Does not modify the original notebook file.
    kernel_name: If None, will try to auto-detect from notebook metadata or use system default.
    """
    import nbformat
    from nbclient import NotebookClient
    from nbformat.v4 import new_code_cell

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    nb = nbformat.read(input_path, as_version=4)

    # Inject a tiny prelude cell to set matplotlib inline and silence non-interactive warnings.
    # This does NOT modify the source notebook; it's only in the executed copy we write out.
    prelude = (
        "# Auto-injected by runner: set inline backend and silence non-interactive warnings\n"
        "%matplotlib inline\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore', message='.*FigureCanvasAgg is non-interactive.*')\n"
    )
    nb.cells.insert(0, new_code_cell(source=prelude))
    
    # Auto-detect kernel name from notebook metadata if not provided
    if kernel_name is None:
        kernel_name = nb.metadata.get('kernelspec', {}).get('name', None)
    
    try:
        client = NotebookClient(nb, timeout=timeout, kernel_name=kernel_name)
        client.execute()
    except Exception as e:
        # If specific kernel fails, try without specifying kernel_name (system default)
        if kernel_name is not None and "No such kernel" in str(e):
            client = NotebookClient(nb, timeout=timeout, kernel_name=None)
            client.execute()
        else:
            raise
    
    nbformat.write(nb, output_path)
    return output_path


def notebook_to_html(notebook_path: str) -> str:
    """Convert an executed notebook to HTML string for preview in Streamlit."""
    import nbformat
    from nbconvert import HTMLExporter

    nb = nbformat.read(notebook_path, as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = False
    (body, _resources) = html_exporter.from_notebook_node(nb)
    return body
