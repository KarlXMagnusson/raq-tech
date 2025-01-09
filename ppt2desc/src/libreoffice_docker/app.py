from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import subprocess
from pathlib import Path
import tempfile
import shutil
import logging

app = FastAPI(title="Document Conversion Service")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

LIBREOFFICE_PATH = Path("/usr/bin/libreoffice")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

@app.post("/convert/ppt-to-pdf")
async def convert_pptx_to_pdf(file: UploadFile):
    """Convert uploaded PPTX file to PDF"""
    logger.info(f"Received file: {file.filename}")

    # Validate file extension
    if not file.filename.lower().endswith(('.pptx', '.ppt')):
        logger.error("Invalid file extension")
        raise HTTPException(status_code=400, detail="File must be a .pptx or .ppt")

    # Create temp dir but don't use context manager
    temp_dir = tempfile.mkdtemp()
    temp_dir_path = Path(temp_dir)
    input_path = temp_dir_path / file.filename

    try:
        # Save uploaded file
        with input_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved uploaded file to: {input_path}")

        # Run LibreOffice conversion
        cmd = [
            str(LIBREOFFICE_PATH),
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', str(temp_dir_path),
            str(input_path)
        ]
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        logger.info(f"LibreOffice stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"LibreOffice stderr: {result.stderr}")

        # Check for output file
        pdf_path = temp_dir_path / f"{input_path.stem}.pdf"
        if not pdf_path.exists():
            logger.error(f"PDF not created. LibreOffice output: {result.stderr}")
            raise HTTPException(status_code=500, detail="PDF conversion failed")

        logger.info(f"Conversion successful: {pdf_path}")

        async def cleanup_background():
            """Async cleanup function"""
            shutil.rmtree(temp_dir, ignore_errors=True)

        response = FileResponse(
            path=pdf_path,
            media_type='application/pdf',
            filename=pdf_path.name
        )
        response.background = cleanup_background
        
        return response

    except Exception as e:
        # Clean up temp dir in case of error
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.exception("Error during conversion")
        raise HTTPException(status_code=500, detail=str(e)) from e
