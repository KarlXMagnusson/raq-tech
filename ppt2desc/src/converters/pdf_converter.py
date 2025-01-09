import logging
import fitz
from PIL import Image
from pathlib import Path
from typing import List
from .exceptions import ConversionError

logger = logging.getLogger(__name__)

def convert_pdf_to_images(pdf_path: Path, temp_dir: Path) -> List[Path]:
    """
    Convert a PDF file to a series of PNG images.
    
    :param pdf_path: Path to the input PDF file
    :param temp_dir: Path to temporary directory for storing images
    :return: List of paths to generated image files
    :raises ConversionError: if the conversion to images fails
    """
    target_size = (1920, 1080)
    image_paths = []
    
    try:
        images_dir = temp_dir / 'images'
        images_dir.mkdir(exist_ok=True)

        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_rect = page.rect
            
            zoom_x = target_size[0] / page_rect.width
            zoom_y = target_size[1] / page_rect.height
            zoom = min(zoom_x, zoom_y)
            
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Create background (white) and paste the rendered image
                new_img = Image.new("RGB", target_size, (255, 255, 255))
                paste_x = (target_size[0] - img.width) // 2
                paste_y = (target_size[1] - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))
                
                # Save image
                image_path = images_dir / f"slide_{page_num + 1}.png"
                new_img.save(image_path)
                image_paths.append(image_path)

            except Exception as inner_exc:
                logger.error(f"Error processing page {page_num + 1}: {str(inner_exc)}")
                continue

        doc.close()
        return image_paths

    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise ConversionError(f"Error converting PDF to images: {str(e)}")
