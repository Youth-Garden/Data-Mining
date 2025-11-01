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


def notebook_to_pdf(notebook_path: str) -> bytes:
    """
    Convert an executed notebook to PDF using ReportLab.
    
    Args:
        notebook_path: Path to the executed notebook (.ipynb)
        
    Returns:
        bytes: PDF file content as bytes
        
    Note:
        Requires: pip install reportlab nbformat
    """
    import nbformat
    from io import BytesIO
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    
    # Read notebook
    nb = nbformat.read(notebook_path, as_version=4)
    
    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#1f1f1f',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#1f1f1f',
        spaceAfter=12,
        spaceBefore=12
    )
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=12
    )
    
    # Add title
    elements.append(Paragraph("Glass Data Mining - Notebook Results", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Process each cell
    for i, cell in enumerate(nb.cells):
        cell_type = cell.cell_type
        
        if cell_type == 'markdown':
            # Add markdown as paragraph
            text = cell.source.replace('\n', '<br/>')
            elements.append(Paragraph(text, styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
            
        elif cell_type == 'code':
            # Add code cell header
            elements.append(Paragraph(f"Cell [{i+1}] (Code):", heading_style))
            
            # Add code
            code_text = cell.source
            elements.append(Preformatted(code_text, code_style))
            
            # Add outputs if any
            if hasattr(cell, 'outputs') and cell.outputs:
                elements.append(Paragraph("Output:", styles['Italic']))
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        output_text = output.text
                        elements.append(Preformatted(output_text, code_style))
                    elif output.output_type == 'execute_result' or output.output_type == 'display_data':
                        if 'text/plain' in output.data:
                            output_text = output.data['text/plain']
                            elements.append(Preformatted(output_text, code_style))
                    elif output.output_type == 'error':
                        error_text = '\n'.join(output.traceback)
                        elements.append(Preformatted(error_text, code_style))
            
            elements.append(Spacer(1, 0.3*inch))
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF bytes
    buffer.seek(0)
    return buffer.getvalue()
