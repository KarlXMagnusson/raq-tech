class LibreOfficeNotFoundError(Exception):
    """Raised when LibreOffice is not found at the given path."""
    pass

class ConversionError(Exception):
    """General error for file conversion issues."""
    pass