import logging
import subprocess
from pathlib import Path
from .exceptions import LibreOfficeNotFoundError, ConversionError

logger = logging.getLogger(__name__)

def convert_pptx_to_pdf(input_file: Path, libreoffice_path: Path, temp_dir: Path) -> Path:
    """
    Convert a PowerPoint file to PDF using LibreOffice.
    
    :param input_file: Path to the input PowerPoint file
    :param libreoffice_path: Path to LibreOffice executable
    :param temp_dir: Temporary directory to store the PDF
    :return: Path to the output PDF file if successful
    :raises LibreOfficeNotFoundError: if LibreOffice is not found
    :raises ConversionError: if the conversion fails
    """
    if not libreoffice_path.exists():
        logger.error(f"LibreOffice not found at {libreoffice_path}")
        raise LibreOfficeNotFoundError(f"LibreOffice not found at {libreoffice_path}")

    try:
        cmd = [
            str(libreoffice_path),
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', str(temp_dir),
            str(input_file)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(f"LibreOffice conversion output: {result.stdout}")

        # The PDF file name should match the PPTX name, but with ".pdf"
        pdf_name = f"{input_file.stem}.pdf"
        pdf_path = temp_dir / pdf_name
        
        if pdf_path.exists():
            return pdf_path
        else:
            logger.error(f"Expected PDF not created at {pdf_path}")
            logger.error(f"LibreOffice error: {result.stderr}")
            raise ConversionError(f"Failed to create PDF at {pdf_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_file}: {e.stderr}")
        raise ConversionError(f"Subprocess conversion error: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error converting {input_file}: {str(e)}")
        raise ConversionError(f"Unexpected error: {str(e)}")
