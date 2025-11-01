def auto_table_height(n_rows: int, row_px: int = 32, min_px: int = 260, max_px: int = 1000) -> int:
    """Compute a suitable table height so most rows are visible without much scrolling."""
    return int(max(min_px, min(max_px, n_rows * row_px + 80)))
