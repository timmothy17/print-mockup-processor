import os
import time
import logging
import traceback
import gc
from PIL import Image
from io import BytesIO
import json

logger = logging.getLogger(__name__)

def extract_blend_info(coordinates_json):
    """
    Extract blend mode information from coordinates JSON.
    Updated to work with new processor structure.
    
    Args:
        coordinates_json: JSON string containing coordinates and blend modes
        
    Returns:
        Tuple of (blend_modes, opacities) dictionaries
    """
    blend_modes = {}
    opacities = {}
    
    try:
        data = json.loads(coordinates_json)
        
        # Check if blend_modes exists in the data
        if 'blend_modes' in data:
            for name, info in data['blend_modes'].items():
                if 'blend_mode' in info:
                    blend_modes[name] = info['blend_mode']
                if 'opacity' in info:
                    opacities[name] = info['opacity']
                    
        logger.info(f"Extracted blend info for {len(blend_modes)} objects")
        
    except Exception as e:
        logger.error(f"Error extracting blend info: {str(e)}")
    
    return blend_modes, opacities

def process_chunk(base_image, overlay_image, opacity=1.0):
    """
    Process a horizontal chunk of the image with proper opacity handling.
    Memory-optimized implementation compatible with new processor.
    
    Args:
        base_image: Base image (PIL Image)
        overlay_image: Overlay image (PIL Image)
        opacity: Opacity of the overlay image (0-1)
        
    Returns:
        Combined image (PIL Image)
    """
    # Both images must be in RGBA mode
    base_is_rgba = base_image.mode == 'RGBA'
    overlay_is_rgba = overlay_image.mode == 'RGBA'
    
    if not base_is_rgba:
        base_image = base_image.convert('RGBA')
    if not overlay_is_rgba:
        overlay_image = overlay_image.convert('RGBA')
    
    # Create result image directly
    result = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
    
    # Paste the base image first
    result.paste(base_image, (0, 0))
    
    # Handle opacity adjustment if needed
    if opacity < 1.0 and opacity > 0:
        # Split the channels
        r, g, b, a = overlay_image.split()
        
        # Adjust the alpha channel by opacity
        a = a.point(lambda x: int(x * opacity))
        
        # Create an intermediate image with adjusted alpha
        overlay_with_opacity = Image.merge('RGBA', (r, g, b, a))
        
        # Paste with the alpha mask
        result.paste(overlay_with_opacity, (0, 0), overlay_with_opacity)
        
        # Clean up intermediate objects
        del r, g, b, a, overlay_with_opacity
    elif opacity > 0:  # No adjustment needed, just paste with full opacity
        result.paste(overlay_image, (0, 0), overlay_image)
    
    # Force garbage collection after pasting large images
    gc.collect()
    
    return result

def create_local_mockup(background_url, warped_artworks, blend_modes=None, opacities=None, 
                        chunk_size=500, record_id=None, intelligent_scaling=True, 
                        original_dimensions=None):
    """
    Create a mockup by layering images locally without using Renderform.
    Updated to work with new processor structure and Photoshop blend modes.
    
    Args:
        background_url: URL to the background image 
        warped_artworks: Dictionary of {layer_name: image_url}
        blend_modes: Dictionary of {layer_name: blend_mode} (for compatibility)
        opacities: Dictionary of {layer_name: opacity} (for compatibility)
        chunk_size: Height of each processing chunk in pixels
        record_id: Airtable record ID for the mockup
        intelligent_scaling: Whether to use intelligent scaling (True) or not (False)
        original_dimensions: Tuple of (width, height) representing original background dimensions
        
    Returns:
        URL to the final mockup on S3
    """
    from s3_handler import S3Handler
    s3_handler = S3Handler()
    
    # Variables that might need cleanup in finally block
    canvas = None
    rgb_canvas = None
    output_buffer = None
    background = None
    
    try:
        # Get current timestamp for filename (fallback if record_id not provided)
        timestamp = int(time.time())
        
        # Check if we have any artwork layers
        if not warped_artworks:
            logger.warning("No warped artworks provided")
            return None
            
        # Determine target output size based on intelligent scaling
        output_size = None
        scale_factor = 1.0
        
        # If original dimensions were provided, use them for scaling decision
        if original_dimensions and intelligent_scaling:
            orig_width, orig_height = original_dimensions
            logger.info(f"Using provided original dimensions: {orig_width}x{orig_height}")
            
            # Determine the maximum dimension from the provided original dimensions
            max_dimension = max(orig_width, orig_height)
            
            # More aggressive scaling based on size to target ~8-10MB final files
            if max_dimension > 6000:
                scale_factor = 0.5  # Extremely large - scale to 50%
                logger.info(f"Extremely large background detected ({orig_width}x{orig_height}), scaling to 50%")
            elif max_dimension > 5000:
                scale_factor = 0.6  # Very large - scale to 60%
                logger.info(f"Very large background detected ({orig_width}x{orig_height}), scaling to 60%")
            elif max_dimension > 4000:
                scale_factor = 0.7  # Large - scale to 70%
                logger.info(f"Large background detected ({orig_width}x{orig_height}), scaling to 70%")
            elif max_dimension > 3000:
                scale_factor = 0.8  # Medium-large - scale to 80%
                logger.info(f"Medium-large background detected ({orig_width}x{orig_height}), scaling to 80%")
            elif max_dimension > 2000:
                scale_factor = 0.9  # Medium - scale to 90%
                logger.info(f"Medium background detected ({orig_width}x{orig_height}), scaling to 90%")
            else:
                scale_factor = 1.0  # Small - keep at 100%
                logger.info(f"Small background detected ({orig_width}x{orig_height}), keeping at original size")
            
            # Calculate target dimensions based on the original dimensions
            target_width = int(orig_width * scale_factor)
            target_height = int(orig_height * scale_factor)
            output_size = (target_width, target_height)
            logger.info(f"Target output size: {output_size} ({scale_factor:.2f}x)")
        
        # Download and extract background metadata only once
        logger.info(f"Loading background: {background_url}")
        background_data = s3_handler.download_file(background_url)
        
        # Check background dimensions without fully loading it
        with Image.open(background_data) as bg_img:
            bg_size = bg_img.size
            bg_format = bg_img.format
            logger.info(f"Downloaded background size: {bg_size}, format: {bg_format}")
            
            # If we didn't have original dimensions, use the downloaded background's dimensions
            if not output_size:
                orig_width, orig_height = bg_size
                
                # Determine appropriate scaling based on image size
                if intelligent_scaling:
                    max_dimension = max(orig_width, orig_height)
                    
                    # Use same scaling logic as above
                    if max_dimension > 6000:
                        scale_factor = 0.5
                        logger.info(f"Extremely large background detected ({orig_width}x{orig_height}), scaling to 50%")
                    elif max_dimension > 5000:
                        scale_factor = 0.6
                        logger.info(f"Very large background detected ({orig_width}x{orig_height}), scaling to 60%")
                    elif max_dimension > 4000:
                        scale_factor = 0.7
                        logger.info(f"Large background detected ({orig_width}x{orig_height}), scaling to 70%")
                    elif max_dimension > 3000:
                        scale_factor = 0.8
                        logger.info(f"Medium-large background detected ({orig_width}x{orig_height}), scaling to 80%")
                    elif max_dimension > 2000:
                        scale_factor = 0.9
                        logger.info(f"Medium background detected ({orig_width}x{orig_height}), scaling to 90%")
                    else:
                        scale_factor = 1.0
                        logger.info(f"Small background detected ({orig_width}x{orig_height}), keeping at original size")
                    
                    # Calculate output size
                    output_width = int(orig_width * scale_factor)
                    output_height = int(orig_height * scale_factor)
                    output_size = (output_width, output_height)
                else:
                    # No intelligent scaling, use original size
                    output_size = bg_size
                    scale_factor = 1.0
                
                logger.info(f"Target output size: {output_size} ({scale_factor:.2f}x)")
        
        # Reset the file pointer
        background_data.seek(0)
        
        # Load the background image once and resize immediately
        background = Image.open(background_data).convert("RGBA")
        
        # Now that we're done with the raw data, free that memory
        del background_data
        gc.collect()
        
        # Resize background to output size if needed - do this only once
        if background.size != output_size:
            logger.info(f"Resizing background from {background.size} to {output_size}")
            
            try:
                # Try using LANCZOS for better quality
                resized_bg = background.resize(output_size, Image.LANCZOS)
                background = resized_bg
            except MemoryError:
                # Fall back to BILINEAR if memory error occurs
                logger.warning("Memory error during background resize. Using lower quality method.")
                try:
                    # Free original image first
                    background = background.resize(output_size, Image.BILINEAR)
                except MemoryError:
                    # Last resort - use NEAREST neighbor, the most memory efficient
                    logger.warning("Second memory error. Using nearest neighbor method.")
                    background = background.resize(output_size, Image.NEAREST)
            
            # Force garbage collection after resize
            gc.collect()
        
        # Create canvas at the output size
        canvas = Image.new('RGBA', output_size, (0, 0, 0, 0))
        width, height = output_size
        
        # ===== LAYER THE WARPED ARTWORKS =====
        
        # Process each artwork layer
        for name, url in warped_artworks.items():
            logger.info(f"Processing layer: {name}")
            
            # Download artwork
            artwork_data = s3_handler.download_file(url)
            artwork = Image.open(artwork_data).convert("RGBA")
            
            # Free artwork_data memory immediately
            del artwork_data
            gc.collect()
            
            # Resize artwork to match our output size if needed
            if artwork.size != output_size:
                logger.info(f"Resizing artwork from {artwork.size} to {output_size}")
                try:
                    # Direct resize attempt
                    artwork_resized = artwork.resize(output_size, Image.LANCZOS)
                    
                    # Replace original with resized
                    del artwork
                    artwork = artwork_resized
                    gc.collect()
                except MemoryError:
                    # Memory-efficient approach for large images
                    logger.info("Using memory-efficient scaling approach due to MemoryError")
                    
                    # Free the original artwork first
                    del artwork
                    gc.collect()
                    
                    # Create a new blank image at the target size
                    artwork = Image.new('RGBA', output_size, (0, 0, 0, 0))
                    
                    # Re-download and resize with a more memory efficient approach
                    temp_artwork_data = s3_handler.download_file(url)
                    temp_artwork = Image.open(temp_artwork_data).convert("RGBA")
                    
                    # Calculate position to center the artwork
                    paste_x = (output_size[0] - temp_artwork.size[0]) // 2
                    paste_y = (output_size[1] - temp_artwork.size[1]) // 2
                    
                    # Paste the original artwork in the center
                    artwork.paste(temp_artwork, (paste_x, paste_y))
                    
                    # Clean up
                    del temp_artwork, temp_artwork_data
                    gc.collect()
            
            # Get opacity for this layer (default to 1.0)
            layer_opacity = 1.0
            if opacities and name in opacities:
                layer_opacity = opacities[name] / 255.0  # Convert from 0-255 to 0-1
            
            # Process in small chunks to minimize memory usage
            for y_start in range(0, height, chunk_size):
                y_end = min(y_start + chunk_size, height)
                
                # Extract chunks
                base_chunk = canvas.crop((0, y_start, width, y_end))
                artwork_chunk = artwork.crop((0, y_start, width, y_end))
                
                # Process chunk
                result_chunk = process_chunk(base_chunk, artwork_chunk, layer_opacity)
                
                # Paste back
                canvas.paste(result_chunk, (0, y_start))
                
                # Free memory
                del base_chunk, artwork_chunk, result_chunk
                gc.collect()
            
            # Free memory
            del artwork
            gc.collect()
        
        # ===== LAYER THE BACKGROUND ON TOP =====
        
        # Process the background in chunks
        for y_start in range(0, height, chunk_size):
            y_end = min(y_start + chunk_size, height)
            
            # Extract chunks
            base_chunk = canvas.crop((0, y_start, width, y_end))
            bg_chunk = background.crop((0, y_start, width, y_end))
            
            # Process chunk
            result_chunk = process_chunk(base_chunk, bg_chunk, 1.0)
            
            # Paste back
            canvas.paste(result_chunk, (0, y_start))
            
            # Free memory
            del base_chunk, bg_chunk, result_chunk
            gc.collect()
        
        # Free memory
        del background
        background = None
        gc.collect()

        # ===== SAVE AND UPLOAD WITH TRANSPARENCY FLATTENED =====
        
        # Flatten transparency by compositing onto a white background
        logger.info("Flattening transparency and converting to high-quality JPEG")
        
        # Create a white background the same size as canvas
        white_background = Image.new('RGB', canvas.size, (255, 255, 255))
        
        # Composite the RGBA canvas onto the white background
        # This flattens all transparency to white
        flattened_canvas = Image.alpha_composite(
            Image.new('RGBA', canvas.size, (48, 47, 47, 255)),  # Opaque white background
            canvas
        ).convert('RGB')  # Convert to RGB since we no longer need alpha
        
        # Free the original canvas
        del canvas
        canvas = None
        gc.collect()
        
        # Save as JPEG with very high quality to minimize compression
        save_format = 'JPEG'
        output_buffer = BytesIO()
        flattened_canvas.save(
            output_buffer, 
            format=save_format, 
            optimize=False,    # Disable optimization to preserve quality
            dpi=(300,300),      # High DPI for print quality
            quality=98,        # Very high quality (95-100 is near-lossless)
            subsampling=0      # Disable chroma subsampling for maximum quality
        )
        
        output_buffer.seek(0)
        
        # Generate output key with correct naming convention
        background_name = os.path.basename(background_url).split('.')[0]
        
        # Include scaling info in filename if not 100%
        scaling_suffix = f"_s{int(scale_factor*100)}" if scale_factor < 1.0 else ""
        
        # Use record_id in the filename if provided
        if record_id:
            output_key = f"ad-mockups/{background_name}_{record_id}{scaling_suffix}_v2.{save_format.lower()}"
        else:
            # Fallback to timestamp
            output_key = f"ad-mockups/{background_name}_{timestamp}{scaling_suffix}_v2.{save_format.lower()}"
        
        # Upload to S3 - this is the critical operation
        logger.info(f"Uploading to S3 as {save_format}: {output_key}")
        mockup_url = s3_handler.upload_file_obj(
            output_buffer,
            output_key,
            content_type=f"image/{save_format.lower()}"
        )
        
        # Log file size
        try:
            if output_buffer and not output_buffer.closed:
                file_size_mb = len(output_buffer.getvalue()) / (1024 * 1024)
                logger.info(f"Final image size: {file_size_mb:.2f}MB")
        except Exception as size_error:
            logger.warning(f"Error calculating file size: {str(size_error)}")
       
       # Successfully got the URL, return it
        
        return mockup_url
       
    except Exception as e:
        # Log the error from the main try block
        logger.error(f"Error creating local mockup: {str(e)}")
        logger.error(traceback.format_exc())
        return None
        
    finally:
        # Final cleanup that always happens
        # Each operation is wrapped in its own try/except to ensure one failure doesn't prevent others
        
        try:
            if output_buffer and hasattr(output_buffer, 'close') and not output_buffer.closed:
                output_buffer.close()
        except Exception as close_error:
            logger.debug(f"Non-critical error during cleanup: {str(close_error)}")
        
        try:
            if canvas:
                del canvas
        except Exception:
            pass
            
        try:
            if rgb_canvas:
                del rgb_canvas
        except Exception:
            pass
        
        try:
            if background:
                del background
        except Exception:
            pass
        
        gc.collect()

def get_memory_usage_mb():
    """Get current memory usage in MB, with fallback if psutil is not available"""
    try:
        # Try to use psutil if available
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # If psutil is not available, return a placeholder value
        return -1  # Indicates memory usage couldn't be determined
