import logging
import traceback
from io import BytesIO
from PIL import Image, ImageEnhance, ImageDraw, ImageStat, ImageOps, ImageChops  
from psd_tools import PSDImage
import cv2
import numpy as np
import json
import time
import gc  # For garbage collection
from s3_handler import S3Handler
from renderform_handler import RenderformHandler

logger = logging.getLogger(__name__)

# Initialize handlers
s3_handler = S3Handler()
renderform_handler = RenderformHandler()

# ==========================================
# GEOMETRIC TRANSFORMATION UTILITIES
# ==========================================

class GeometricProcessor:
    """Handles all geometric transformations and coordinate calculations."""
    
    @staticmethod
    def calculate_frame_region_with_padding(transform_points, padding=4):
        """
        Calculate the bounding box of the frame region with padding.
        
        Args:
            transform_points: List of (x,y) tuples defining the frame
            padding: Fixed padding in pixels around the frame region
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y) defining the region to process
        """
        try:
            # Ensure transform_points are in the right format
            points = []
            for point in transform_points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))
                else:
                    raise ValueError(f"Invalid point format: {point}")
            
            if len(points) < 3:
                raise ValueError("Need at least 3 points to define a frame region")
                
            # Calculate bounding box using vectorized operations
            points_array = np.array(points)
            min_coords = np.floor(np.min(points_array, axis=0)).astype(int)
            max_coords = np.ceil(np.max(points_array, axis=0)).astype(int)
            
            min_x = max(0, min_coords[0] - padding)
            min_y = max(0, min_coords[1] - padding)
            max_x = max_coords[0] + padding
            max_y = max_coords[1] + padding
            
            logger.info(f"Frame region with {padding}px padding: ({min_x}, {min_y}, {max_x}, {max_y})")
            
            return (min_x, min_y, max_x, max_y)
            
        except Exception as e:
            logger.error(f"Error calculating frame region: {str(e)}")
            raise

    @staticmethod
    def apply_perspective_transform(image, src_points, dst_points, output_size):
        """Apply perspective transform to an image with proper alpha handling"""
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Determine if the image has an alpha channel
        has_alpha = len(img_array.shape) == 3 and img_array.shape[2] == 4
        
        if has_alpha:
            # Split the image into color and alpha channels
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
        else:
            # If no alpha, just use the RGB channels and create a full opacity alpha
            rgb = img_array
            alpha = np.full((img_array.shape[0], img_array.shape[1]), 255, dtype=np.uint8)
        
        # Convert points to numpy arrays
        src_np = np.array(src_points, dtype=np.float32)
        dst_np = np.array(dst_points, dtype=np.float32)
        
        # Calculate the perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(src_np, dst_np)
        
        # Warp the color channels
        warped_rgb = cv2.warpPerspective(
            rgb, 
            transform_matrix, 
            output_size, 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[0, 0, 0]
        )
        
        # Warp the alpha channel separately
        warped_alpha = cv2.warpPerspective(
            alpha, 
            transform_matrix, 
            output_size, 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0  # Transparent border
        )
        
        # Combine RGB and alpha back together
        warped_rgba = np.zeros((output_size[1], output_size[0], 4), dtype=np.uint8)
        warped_rgba[:, :, :3] = warped_rgb
        warped_rgba[:, :, 3] = warped_alpha
        
        # Convert back to PIL image
        return Image.fromarray(warped_rgba)

    @staticmethod
    def smooth_edges(image, kernel_size=3):
        """Apply light Gaussian blur to smooth jagged edges"""
        image_array = np.array(image)
        smoothed = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0.5)
        return Image.fromarray(smoothed)

    @staticmethod
    def calculate_target_dimensions(transform_points):
        """Calculate target dimensions from transform points"""
        if isinstance(transform_points, list) and transform_points and isinstance(transform_points[0], list):
            transform_points = [(point[0], point[1]) for point in transform_points]
        
        # Use NumPy for coordinate calculations
        coords_array = np.array(transform_points)
        min_coords = np.min(coords_array, axis=0)
        max_coords = np.max(coords_array, axis=0)
        
        target_width = int(max_coords[0] - min_coords[0])
        target_height = int(max_coords[1] - min_coords[1])
        
        return target_width, target_height

# ==========================================
# PHOTOSHOP-STYLE BLEND MODES
# ==========================================

class PhotoshopBlendModes:
    """Vectorized implementations of Photoshop blend modes."""
    
    @staticmethod
    def multiply_blend(top_rgba, bottom_rgba):
        """
        Apply Photoshop multiply blend mode using fully vectorized NumPy operations.
        
        Args:
            top_rgba: Top layer as numpy array (H, W, 4) with values 0-255
            bottom_rgba: Bottom layer as numpy array (H, W, 4) with values 0-255
            
        Returns:
            numpy array (H, W, 4) with blend applied
        """
        # Convert to float32 for calculations (more memory efficient than float64)
        top = top_rgba.astype(np.float32)
        bottom = bottom_rgba.astype(np.float32)
        
        # Normalize to 0-1 range
        top_norm = top / 255.0
        bottom_norm = bottom / 255.0
        
        # Split RGB and alpha channels
        top_rgb = top_norm[:, :, :3]
        top_alpha = top_norm[:, :, 3]
        bottom_rgb = bottom_norm[:, :, :3]
        bottom_alpha = bottom_norm[:, :, 3]
        
        # Photoshop multiply formula: (top * bottom) for RGB - fully vectorized
        multiply_rgb = top_rgb * bottom_rgb
        
        # Standard alpha compositing: out_alpha = top_alpha + bottom_alpha * (1 - top_alpha)
        result_alpha = top_alpha + bottom_alpha * (1.0 - top_alpha)
        
        # Alpha composite the RGB channels - vectorized
        # Expand alpha dimensions for broadcasting
        top_alpha_expanded = top_alpha[:, :, np.newaxis]
        bottom_alpha_expanded = bottom_alpha[:, :, np.newaxis]
        result_alpha_expanded = result_alpha[:, :, np.newaxis]
        
        # Avoid division by zero
        safe_result_alpha = np.where(result_alpha_expanded > 0.0, result_alpha_expanded, 1.0)
        
        # Composite RGB: (multiply_rgb * top_alpha + bottom_rgb * bottom_alpha * (1 - top_alpha)) / result_alpha
        numerator = (multiply_rgb * top_alpha_expanded + 
                    bottom_rgb * bottom_alpha_expanded * (1.0 - top_alpha_expanded))
        
        result_rgb = np.where(
            result_alpha_expanded > 0.0,
            numerator / safe_result_alpha,
            0.0
        )
        
        # Combine RGB and alpha back together
        result = np.zeros_like(top_norm)
        result[:, :, :3] = result_rgb
        result[:, :, 3] = result_alpha
        
        # Convert back to 0-255 range and uint8 - vectorized
        result_uint8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        
        return result_uint8

    @staticmethod
    def luminosity_blend(base_rgba, blend_rgba, opacity=1.0):
        """
        Apply Photoshop luminosity blend mode using fully vectorized NumPy operations.
        
        Args:
            base_rgba: Base layer as numpy array (H, W, 4) with values 0-255
            blend_rgba: Blend layer as numpy array (H, W, 4) with values 0-255
            opacity: Blend opacity (0.0-1.0)
            
        Returns:
            numpy array (H, W, 4) with blend applied
        """
        # Convert to float32 for calculations
        base = base_rgba.astype(np.float32) / 255.0
        blend = blend_rgba.astype(np.float32) / 255.0
        
        # Apply opacity to blend layer
        blend[:, :, 3] *= opacity
        
        # Split channels
        base_rgb = base[:, :, :3]
        base_alpha = base[:, :, 3]
        blend_rgb = blend[:, :, :3]
        blend_alpha = blend[:, :, 3]
        
        # Calculate luminosity using ITU-R BT.601 weights - vectorized
        # Weights as numpy array for efficient computation
        lum_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        
        # Calculate luminosity for both layers - vectorized dot product
        base_lum = np.dot(base_rgb, lum_weights)
        blend_lum = np.dot(blend_rgb, lum_weights)
        
        # Luminosity difference
        lum_diff = blend_lum[:, :, np.newaxis] - base_lum[:, :, np.newaxis]
        
        # Apply luminosity blend: adjust base RGB by luminosity difference
        result_rgb = base_rgb + lum_diff
        
        # Clamp RGB values to valid range
        result_rgb = np.clip(result_rgb, 0.0, 1.0)
        
        # Alpha compositing - vectorized
        blend_alpha_expanded = blend_alpha[:, :, np.newaxis]
        base_alpha_expanded = base_alpha[:, :, np.newaxis]
        
        result_alpha = blend_alpha + base_alpha * (1.0 - blend_alpha)
        result_alpha_expanded = result_alpha[:, :, np.newaxis]
        
        # Avoid division by zero
        safe_result_alpha = np.where(result_alpha_expanded > 0.0, result_alpha_expanded, 1.0)
        
        # Composite RGB
        numerator = (result_rgb * blend_alpha_expanded + 
                    base_rgb * base_alpha_expanded * (1.0 - blend_alpha_expanded))
        
        final_rgb = np.where(
            result_alpha_expanded > 0.0,
            numerator / safe_result_alpha,
            0.0
        )
        
        # Combine results
        result = np.zeros_like(base)
        result[:, :, :3] = final_rgb
        result[:, :, 3] = result_alpha
        
        # Convert back to uint8 - vectorized
        result_uint8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        
        return result_uint8

    @staticmethod
    def normal_blend(top_rgba, bottom_rgba, opacity=1.0):
        """
        Apply normal blend mode using vectorized operations.
        
        Args:
            top_rgba: Top layer as numpy array (H, W, 4)
            bottom_rgba: Bottom layer as numpy array (H, W, 4)
            opacity: Blend opacity (0.0-1.0)
            
        Returns:
            numpy array with normal blend applied
        """
        # Convert to float32 for calculations
        top = top_rgba.astype(np.float32) / 255.0
        bottom = bottom_rgba.astype(np.float32) / 255.0
        
        # Apply opacity to top layer
        top[:, :, 3] *= opacity
        
        # Standard alpha compositing - fully vectorized
        top_alpha = top[:, :, 3:4]  # Keep as 4D for broadcasting
        
        # Alpha blend: result = top * alpha + bottom * (1 - alpha)
        result = top * top_alpha + bottom * (1.0 - top_alpha)
        
        # Alpha compositing for alpha channel
        result[:, :, 3] = top[:, :, 3] + bottom[:, :, 3] * (1.0 - top[:, :, 3])
        
        # Convert back to uint8
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)

# ==========================================
# TILED BLENDING PROCESSOR
# ==========================================

class TiledBlendProcessor:
    """Handles memory-efficient tiled blending operations."""
    
    def __init__(self, blend_modes_handler=None):
        self.blend_modes = blend_modes_handler or PhotoshopBlendModes()
    
    def apply_blend_mode_to_tile(self, top_tile, bottom_tile, blend_mode='multiply', opacity=1.0, artwork_mask=None):
        """
        Apply specified blend mode to image tiles using vectorized operations with masking.

        Args:
            top_tile: Top layer tile as PIL Image (RGBA)
            bottom_tile: Bottom layer tile as PIL Image (RGBA)
            blend_mode: Blend mode ('multiply', 'luminosity', etc.)
            opacity: Blend opacity (0.0-1.0)
            artwork_mask: Optional PIL Image mask (L mode) to limit blend area
            
        Returns:
            PIL Image with blend applied only in masked areas
        """
        # Convert PIL images to numpy arrays - single operation, no copying
        top_array = np.asarray(top_tile, dtype=np.uint8)
        bottom_array = np.asarray(bottom_tile, dtype=np.uint8)
        
        # Ensure both are RGBA (should be guaranteed by caller)
        if top_array.shape[2] != 4 or bottom_array.shape[2] != 4:
            raise ValueError("Both tiles must be RGBA format")
        
        # Apply the specified blend mode using vectorized functions
        if blend_mode.lower() == 'multiply':
            blended_array = self.blend_modes.multiply_blend(top_array, bottom_array)
        elif blend_mode.lower() == 'luminosity':
            blended_array = self.blend_modes.luminosity_blend(bottom_array, top_array, opacity)
        else:
            # Normal blend mode - vectorized
            logger.warning(f"Unknown blend mode '{blend_mode}', using normal blend")
            blended_array = self.blend_modes.normal_blend(top_array, bottom_array, opacity)
        
        # Apply artwork mask if provided
        if artwork_mask is not None:
            mask_array = np.asarray(artwork_mask, dtype=np.uint8)
            
            # Ensure mask is single channel
            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, 0]  # Take first channel if RGB
            
            # Normalize mask to 0-1 range
            mask_normalized = mask_array.astype(np.float32) / 255.0
            mask_expanded = mask_normalized[:, :, np.newaxis]  # Add channel dimension for broadcasting
            
            # Convert arrays to float for blending
            top_float = top_array.astype(np.float32)
            blended_float = blended_array.astype(np.float32)
            
            # Apply mask: blended where mask=1, transparent where mask=0
            # RGB channels: use blended where mask is white, transparent where mask is black
            result_rgb = blended_float[:, :, :3] * mask_expanded
            
            # Alpha channel: use artwork's alpha where mask is white, 0 where mask is black
            result_alpha = top_float[:, :, 3] * mask_normalized
            
            # Combine RGB and alpha
            result_array = np.zeros_like(blended_float)
            result_array[:, :, :3] = result_rgb
            result_array[:, :, 3] = result_alpha
            
            # Convert back to uint8
            result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        else:
            # No mask, use full blended result
            result_array = blended_array
        
        # Convert back to PIL Image - single operation
        return Image.fromarray(result_array, mode='RGBA')
    
    def blend_images_tiled(self, top_image_url, bottom_image_url, frame_region, 
                        blend_mode='multiply', tile_size=(128, 128), 
                        opacity=1.0, padding=4):
        """
        Memory-efficient tiled blending with automatic masking to prevent white areas.
        """
        # Variables for cleanup
        top_data = None
        bottom_data = None
        top_image = None
        bottom_image = None
        result_image = None
        artwork_mask = None
        
        try:
            # Download and load images efficiently
            logger.info(f"Downloading images for tiled blending")
            
            # Download both images
            top_data = s3_handler.download_file(top_image_url)
            bottom_data = s3_handler.download_file(bottom_image_url)
            
            # Load images as RGBA
            top_image = Image.open(top_data).convert('RGBA')
            bottom_image = Image.open(bottom_data).convert('RGBA')
            
            # Free download data immediately
            del top_data, bottom_data
            top_data = bottom_data = None
            gc.collect()
            
            # Ensure images are the same size
            if top_image.size != bottom_image.size:
                logger.info(f"Resizing top image from {top_image.size} to {bottom_image.size}")
                top_image = top_image.resize(bottom_image.size, Image.LANCZOS)
            
            # Extract and validate frame region
            min_x, min_y, max_x, max_y = frame_region
            img_width, img_height = bottom_image.size
            
            # Clamp region to image bounds
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(img_width, max_x)
            max_y = min(img_height, max_y)
            
            region_width = max_x - min_x
            region_height = max_y - min_y
            
            if region_width <= 0 or region_height <= 0:
                logger.warning("Invalid frame region, returning transparent image")
                return Image.new('RGBA', bottom_image.size, (0, 0, 0, 0))
            
            logger.info(f"Processing region: ({min_x}, {min_y}, {max_x}, {max_y}) - {region_width}x{region_height}")
            logger.info(f"Using {blend_mode} blend mode with opacity {opacity:.2f}")
            
            # CREATE ARTWORK MASK: Only blend in the frame region
            artwork_mask = Image.new('L', bottom_image.size, 0)  # Start with black (transparent)
            mask_draw = ImageDraw.Draw(artwork_mask)
            
            # Fill the frame region with white (where blend should happen)
            mask_draw.rectangle([(min_x, min_y), (max_x, max_y)], fill=255)
            
            # Optional: Make mask more precise by using artwork's alpha channel
            # Extract alpha from artwork and use it to refine the mask
            artwork_alpha = top_image.split()[-1]  # Get alpha channel
            
            # Combine rectangular mask with artwork alpha for precise masking
            combined_mask = Image.new('L', bottom_image.size, 0)
            combined_mask.paste(artwork_alpha, (0, 0), artwork_alpha)  # Paste artwork alpha
            
            # Intersect with rectangular region mask
            combined_mask = ImageChops.multiply(combined_mask, artwork_mask)
            artwork_mask = combined_mask
            
            logger.info(f"Created artwork mask for precise blending")
            
            # Create result image starting with transparent
            result_image = Image.new('RGBA', bottom_image.size, (0, 0, 0, 0))
            
            # Process tiles within the frame region with masking
            tile_width, tile_height = tile_size
            tiles_processed = 0
            total_tiles = ((max_x - min_x + tile_width - 1) // tile_width) * \
                        ((max_y - min_y + tile_height - 1) // tile_height)
            
            logger.info(f"Processing {total_tiles} tiles of size {tile_size} with artwork masking")
            
            # Process tiles using vectorized operations with masking
            for y in range(min_y, max_y, tile_height):
                for x in range(min_x, max_x, tile_width):
                    # Calculate actual tile boundaries
                    tile_min_x = x
                    tile_min_y = y
                    tile_max_x = min(x + tile_width, max_x)
                    tile_max_y = min(y + tile_height, max_y)
                    
                    # Extract tiles - single crop operation each
                    top_tile = top_image.crop((tile_min_x, tile_min_y, tile_max_x, tile_max_y))
                    bottom_tile = bottom_image.crop((tile_min_x, tile_min_y, tile_max_x, tile_max_y))
                    mask_tile = artwork_mask.crop((tile_min_x, tile_min_y, tile_max_x, tile_max_y))
                    
                    # Apply vectorized blend mode with masking
                    blended_tile = self.apply_blend_mode_to_tile(
                        top_tile, bottom_tile, blend_mode, opacity, mask_tile
                    )
                    
                    # Paste result back - single paste operation
                    result_image.paste(blended_tile, (tile_min_x, tile_min_y), blended_tile)
                    
                    tiles_processed += 1
                    
                    # Clean up tile memory immediately
                    del top_tile, bottom_tile, mask_tile, blended_tile
                    
                    # Periodic garbage collection for memory management
                    if tiles_processed % 20 == 0:  # Less frequent GC
                        gc.collect()
                        if tiles_processed % 100 == 0:  # Progress logging
                            progress = (tiles_processed / total_tiles) * 100
                            logger.info(f"Processed {tiles_processed}/{total_tiles} tiles ({progress:.1f}%)")
            
            logger.info(f"Completed processing {tiles_processed} tiles using masked {blend_mode} blend mode")
            
            return result_image
            
        except Exception as e:
            logger.error(f"Error in vectorized tiled blending with masking: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        finally:
            # Comprehensive cleanup
            cleanup_objects = [top_data, bottom_data, top_image, bottom_image, artwork_mask]
            for obj in cleanup_objects:
                try:
                    if obj is not None:
                        del obj
                except:
                    pass
            gc.collect()
# ==========================================
# PHOTOSHOP-STYLE COLOR ADJUSTMENTS
# ==========================================

class PhotoshopColorAdjustments:
    """Handles Photoshop-style color adjustments with -100 to +100 scale."""
    
    def create_adjustments_json(smart_object_names, default_adjustments=None):
        """
        Create JSON structure for Photoshop-style color adjustments (-100 to +100 scale).
        
        Args:
            smart_object_names: List of smart object names to include
            default_adjustments: Optional default values to use
        
        Returns:
            JSON string with adjustment parameters
        """
        # Default adjustment values (0 = no change)
        if default_adjustments is None:
            default_adjustments = {
                "brightness": 0,      # -100 to +100 (-100 = black, +100 = white)
                "contrast": 0,        # -100 to +100 (-100 = flat gray, +100 = maximum contrast)
                "saturation": 0,      # -100 to +100 (-100 = grayscale, +100 = hyper-saturated)
                "vibrance": 0,        # -100 to +100 (smart saturation adjustment)
                "highlights": 0,      # -100 to +100 (adjust bright areas)
                "shadows": 0,         # -100 to +100 (adjust dark areas)
                "whites": 0,          # -100 to +100 (adjust white point)
                "blacks": 0,          # -100 to +100 (adjust black point)
                "exposure": 0,        # -100 to +100 (overall exposure adjustment)
                "gamma": 0,           # -100 to +100 (midtone adjustment)
                "hue_shift": 0,       # -100 to +100 (hue rotation)
                "warmth": 0,          # -100 to +100 (color temperature)
                "tint": 0,            # -100 to +100 (green-magenta shift)
                "clarity": 0,         # -100 to +100 (midtone contrast)
                "structure": 0        # -100 to +100 (fine detail enhancement)
            }
        
        adjustments_data = {
            "_metadata": {
                "description": "Photoshop-style color adjustment parameters (applied after blend modes)",
                "version": "2.0",
                "scale": "All values range from -100 to +100, where 0 = no adjustment",
                "adjustments": {
                    "brightness": "Overall brightness (-100=black, 0=original, +100=white)",
                    "contrast": "Overall contrast (-100=flat, 0=original, +100=maximum)",
                    "saturation": "Color intensity (-100=grayscale, 0=original, +100=hyper-saturated)",
                    "vibrance": "Smart saturation (protects skin tones, -100 to +100)",
                    "highlights": "Bright area adjustment (-100=darker highlights, +100=brighter)",
                    "shadows": "Dark area adjustment (-100=darker shadows, +100=brighter)",
                    "whites": "White point adjustment (-100=gray whites, +100=pure white)",
                    "blacks": "Black point adjustment (-100=pure black, +100=gray blacks)",
                    "exposure": "Exposure compensation (-100=underexposed, +100=overexposed)",
                    "gamma": "Midtone adjustment (-100=darker mids, +100=brighter mids)",
                    "hue_shift": "Hue rotation (-100 to +100, affects all colors)",
                    "warmth": "Color temperature (-100=cooler/blue, +100=warmer/orange)",
                    "tint": "Green-magenta shift (-100=green, +100=magenta)",
                    "clarity": "Midtone contrast (-100=soft, +100=crisp)",
                    "structure": "Fine detail enhancement (-100=smooth, +100=detailed)"
                },
                "application_order": [
                    "exposure", "highlights", "shadows", "whites", "blacks",
                    "brightness", "contrast", "gamma", "clarity", "structure",
                    "saturation", "vibrance", "hue_shift", "warmth", "tint"
                ]
            },
            "_global_settings": {
                "luminosity_blend_strength": 20,  # NEW: Global luminosity blend (0-100)
                "_description": "Global settings that apply to all artworks during processing"
            }
        }
        
        # Add default adjustments for each smart object
        for obj_name in smart_object_names:
            adjustments_data[obj_name] = default_adjustments.copy()
        
        return json.dumps(adjustments_data, indent=2)

    @staticmethod
    def parse_adjustments(adjustments_json, smart_object_name):
        """
        Parse Photoshop-style adjustments from JSON for a specific smart object.
        Also extracts global settings.
        
        Args:
            adjustments_json: JSON string with manual adjustments
            smart_object_name: Name of the smart object to get adjustments for
        
        Returns:
            Dictionary with adjustment values or None if not found
        """
        try:
            if not adjustments_json or adjustments_json.strip() == "":
                return None
                
            data = json.loads(adjustments_json)
            
            # Skip metadata and get the smart object adjustments
            if smart_object_name in data and smart_object_name not in ["_metadata", "_global_settings"]:
                adjustments = data[smart_object_name]
                logger.info(f"Using Photoshop-style adjustments for {smart_object_name}: {adjustments}")
                return adjustments
            else:
                logger.warning(f"No adjustments found for {smart_object_name}")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing adjustments JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading adjustments: {e}")
            return None
        
    @staticmethod
    def parse_global_settings(adjustments_json):
        """
        Parse global settings from adjustments JSON.
        
        Args:
            adjustments_json: JSON string with adjustments
            
        Returns:
            Dictionary with global settings
        """
        try:
            if not adjustments_json or adjustments_json.strip() == "":
                return {"luminosity_blend_strength": 20}  # Default
                
            data = json.loads(adjustments_json)
            
            # Get global settings
            if "_global_settings" in data:
                global_settings = data["_global_settings"]
                logger.info(f"Using global settings: {global_settings}")
                return global_settings
            else:
                logger.info("No global settings found, using defaults")
                return {"luminosity_blend_strength": 20}  # Default
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing global settings JSON: {e}")
            return {"luminosity_blend_strength": 20}  # Default
        except Exception as e:
            logger.error(f"Error reading global settings: {e}")
            return {"luminosity_blend_strength": 20}  # Default


    @staticmethod
    def convert_scale_to_factor(value, adjustment_type):
        """
        Convert Photoshop-style -100 to +100 scale to multiplication factors.
        
        Args:
            value: Adjustment value (-100 to +100)
            adjustment_type: Type of adjustment for specific scaling
        
        Returns:
            Multiplication factor for the adjustment
        """
        # Clamp value to valid range
        value = max(-100, min(100, value))
        
        if adjustment_type in ['brightness', 'exposure']:
            # Linear scaling: -100 = 0.0, 0 = 1.0, +100 = 2.0
            return 1.0 + (value / 100.0)
        
        elif adjustment_type in ['contrast', 'clarity', 'structure']:
            # Exponential scaling for contrast: -100 = 0.1, 0 = 1.0, +100 = 3.0
            if value == 0:
                return 1.0
            elif value > 0:
                return 1.0 + (value / 100.0) * 2.0  # 1.0 to 3.0
            else:
                return 1.0 + (value / 100.0) * 0.9  # 0.1 to 1.0
        
        elif adjustment_type in ['saturation', 'vibrance']:
            # Saturation scaling: -100 = 0.0, 0 = 1.0, +100 = 2.5
            if value == 0:
                return 1.0
            elif value > 0:
                return 1.0 + (value / 100.0) * 1.5  # 1.0 to 2.5
            else:
                return 1.0 + (value / 100.0)  # 0.0 to 1.0
        
        elif adjustment_type == 'gamma':
            # Gamma scaling: -100 = 0.3, 0 = 1.0, +100 = 3.0
            if value == 0:
                return 1.0
            elif value > 0:
                return 1.0 + (value / 100.0) * 2.0  # 1.0 to 3.0
            else:
                return 1.0 + (value / 100.0) * 0.7  # 0.3 to 1.0
        
        elif adjustment_type in ['hue_shift']:
            # Hue shift in degrees: -100 = -180°, 0 = 0°, +100 = +180°
            return value * 1.8  # Convert to degrees
        
        elif adjustment_type in ['warmth', 'tint']:
            # Color temperature/tint scaling: -100 to +100 as percentage
            return value / 100.0
        
        elif adjustment_type in ['highlights', 'shadows', 'whites', 'blacks']:
            # Tone adjustments: -100 to +100 as factors
            return 1.0 + (value / 100.0) * 0.5  # 0.5 to 1.5 range
        
        else:
            # Default linear scaling
            return 1.0 + (value / 100.0)

    def apply_adjustments_vectorized(self, image, adjustments):
        """
        Apply Photoshop-style color adjustments using vectorized operations.
        
        Args:
            image: PIL Image to adjust
            adjustments: Dictionary with adjustment values (-100 to +100)
        
        Returns:
            PIL Image with adjustments applied
        """
        try:
            if not adjustments:
                return image
            
            # Convert image to numpy array for vectorized processing
            img_array = np.array(image).astype(np.float32)
            
            # Separate RGB and alpha channels
            if img_array.shape[2] == 4:  # RGBA
                rgb_array = img_array[:, :, :3]
                alpha_array = img_array[:, :, 3]
                has_alpha = True
            else:  # RGB
                rgb_array = img_array
                alpha_array = None
                has_alpha = False
            
            # Normalize RGB to 0-1 range
            rgb_normalized = rgb_array / 255.0
            
            # Apply adjustments in Photoshop order
            adjustment_order = [
                'exposure', 'highlights', 'shadows', 'whites', 'blacks',
                'brightness', 'contrast', 'gamma', 'brightness', 'contrast', 'gamma', 'clarity', 'structure',
                'saturation', 'vibrance', 'hue_shift', 'warmth', 'tint'
            ]
            
            for adj_type in adjustment_order:
                if adj_type in adjustments and adjustments[adj_type] != 0:
                    value = adjustments[adj_type]
                    logger.debug(f"Applying {adj_type}: {value}")
                    
                    if adj_type == 'exposure':
                        # Exposure adjustment (affects all channels equally)
                        factor = self.convert_scale_to_factor(value, adj_type)
                        rgb_normalized = np.clip(rgb_normalized * factor, 0.0, 1.0)
                    
                    elif adj_type == 'brightness':
                        # Brightness adjustment (linear)
                        offset = value / 100.0  # -1.0 to +1.0
                        rgb_normalized = np.clip(rgb_normalized + offset, 0.0, 1.0)
                    
                    elif adj_type == 'contrast':
                        # Contrast adjustment around midpoint
                        factor = self.convert_scale_to_factor(value, adj_type)
                        rgb_normalized = np.clip(((rgb_normalized - 0.5) * factor) + 0.5, 0.0, 1.0)
                    
                    elif adj_type == 'gamma':
                        # Gamma correction
                        gamma = self.convert_scale_to_factor(value, adj_type)
                        rgb_normalized = np.power(rgb_normalized, 1.0 / gamma)
                    
                    elif adj_type == 'highlights':
                        # Adjust highlights (bright areas)
                        factor = self.convert_scale_to_factor(value, adj_type)
                        # Create highlight mask (areas > 0.7)
                        highlight_mask = rgb_normalized > 0.7
                        rgb_normalized = np.where(highlight_mask, 
                                                np.clip(rgb_normalized * factor, 0.0, 1.0),
                                                rgb_normalized)
                    
                    elif adj_type == 'shadows':
                        # Adjust shadows (dark areas)
                        factor = self.convert_scale_to_factor(value, adj_type)
                        # Create shadow mask (areas < 0.3)
                        shadow_mask = rgb_normalized < 0.3
                        rgb_normalized = np.where(shadow_mask,
                                                np.clip(rgb_normalized * factor, 0.0, 1.0),
                                                rgb_normalized)
                    
                    elif adj_type == 'whites':
                        # Adjust white point
                        factor = self.convert_scale_to_factor(value, adj_type)
                        # Affect very bright areas (> 0.9)
                        white_mask = rgb_normalized > 0.9
                        rgb_normalized = np.where(white_mask,
                                                np.clip(rgb_normalized * factor, 0.0, 1.0),
                                                rgb_normalized)
                    
                    elif adj_type == 'blacks':
                        # Adjust black point
                        factor = self.convert_scale_to_factor(value, adj_type)
                        # Affect very dark areas (< 0.1)
                        black_mask = rgb_normalized < 0.1
                        rgb_normalized = np.where(black_mask,
                                                np.clip(rgb_normalized * factor, 0.0, 1.0),
                                                rgb_normalized)
                    
                    elif adj_type in ['saturation', 'vibrance']:
                        # Saturation/vibrance adjustment
                        factor = self.convert_scale_to_factor(value, adj_type)
                        
                        # Convert to grayscale for desaturation
                        gray_weights = np.array([0.299, 0.587, 0.114])
                        gray = np.dot(rgb_normalized, gray_weights)
                        gray_expanded = gray[:, :, np.newaxis]
                        
                        # Blend between grayscale and original based on factor
                        if factor < 1.0:
                            # Desaturation
                            rgb_normalized = gray_expanded + (rgb_normalized - gray_expanded) * factor
                        else:
                            # Saturation boost
                            rgb_normalized = rgb_normalized + (rgb_normalized - gray_expanded) * (factor - 1.0)
                        
                        rgb_normalized = np.clip(rgb_normalized, 0.0, 1.0)
                    
                    elif adj_type == 'hue_shift':
                        # Hue shift (simplified RGB rotation)
                        degrees = self.convert_scale_to_factor(value, adj_type)
                        # This is a simplified hue shift - for full HSV conversion, use colorsys
                        # For now, apply a simple color channel rotation
                        if abs(degrees) > 1:  # Only apply if significant
                            # Simplified hue rotation matrix (approximate)
                            cos_h = np.cos(np.radians(degrees))
                            sin_h = np.sin(np.radians(degrees))
                            
                            # Apply rotation to RGB channels
                            r, g, b = rgb_normalized[:, :, 0], rgb_normalized[:, :, 1], rgb_normalized[:, :, 2]
                            new_r = r * cos_h - g * sin_h
                            new_g = r * sin_h + g * cos_h
                            new_b = b  # Blue channel less affected in this simplified version
                            
                            rgb_normalized = np.stack([new_r, new_g, new_b], axis=-1)
                            rgb_normalized = np.clip(rgb_normalized, 0.0, 1.0)
                    
                    elif adj_type == 'warmth':
                        # Color temperature adjustment
                        warmth_factor = self.convert_scale_to_factor(value, adj_type)
                        if warmth_factor > 0:  # Warmer (more orange/red)
                            rgb_normalized[:, :, 0] *= (1.0 + warmth_factor * 0.2)  # Boost red
                            rgb_normalized[:, :, 1] *= (1.0 + warmth_factor * 0.1)  # Slight green boost
                            rgb_normalized[:, :, 2] *= (1.0 - warmth_factor * 0.2)  # Reduce blue
                        else:  # Cooler (more blue)
                            rgb_normalized[:, :, 0] *= (1.0 + warmth_factor * 0.2)  # Reduce red
                            rgb_normalized[:, :, 1] *= (1.0 + warmth_factor * 0.1)  # Slight green reduction
                            rgb_normalized[:, :, 2] *= (1.0 - warmth_factor * 0.2)  # Boost blue
                        
                        rgb_normalized = np.clip(rgb_normalized, 0.0, 1.0)
                    
                    elif adj_type == 'tint':
                        # Green-magenta tint adjustment
                        tint_factor = self.convert_scale_to_factor(value, adj_type)
                        if tint_factor > 0:  # More magenta
                            rgb_normalized[:, :, 0] *= (1.0 + tint_factor * 0.1)  # Boost red slightly
                            rgb_normalized[:, :, 1] *= (1.0 - tint_factor * 0.2)  # Reduce green
                            rgb_normalized[:, :, 2] *= (1.0 + tint_factor * 0.1)  # Boost blue slightly
                        else:  # More green
                            rgb_normalized[:, :, 0] *= (1.0 + tint_factor * 0.1)  # Reduce red slightly
                            rgb_normalized[:, :, 1] *= (1.0 - tint_factor * 0.2)  # Boost green
                            rgb_normalized[:, :, 2] *= (1.0 + tint_factor * 0.1)  # Reduce blue slightly
                        
                        rgb_normalized = np.clip(rgb_normalized, 0.0, 1.0)
                    
                    elif adj_type in ['clarity', 'structure']:
                        # Clarity/structure (midtone contrast enhancement)
                        factor = self.convert_scale_to_factor(value, adj_type)
                        if factor != 1.0:
                            # Create midtone mask (areas between 0.2 and 0.8)
                            midtone_mask = (rgb_normalized > 0.2) & (rgb_normalized < 0.8)
                            # Apply contrast enhancement to midtones
                            enhanced = np.clip(((rgb_normalized - 0.5) * factor) + 0.5, 0.0, 1.0)
                            rgb_normalized = np.where(midtone_mask, enhanced, rgb_normalized)
            
            # Convert back to 0-255 range
            rgb_final = np.clip(rgb_normalized * 255.0, 0, 255).astype(np.uint8)
            
            # Recombine with alpha channel if present
            if has_alpha:
                result_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
                result_array[:, :, :3] = rgb_final
                result_array[:, :, 3] = alpha_array.astype(np.uint8)
                result_image = Image.fromarray(result_array, 'RGBA')
            else:
                result_image = Image.fromarray(rgb_final, 'RGB')
            
            logger.info(f"Applied {len([k for k, v in adjustments.items() if v != 0])} color adjustments")
            
            return result_image
            
        except Exception as e:
            logger.error(f"Error applying Photoshop-style adjustments: {str(e)}")
            logger.error(traceback.format_exc())
            return image

# ==========================================
# ARTWORK PROCESSING PIPELINE
# ==========================================

class ArtworkProcessor:
    """Complete artwork processing pipeline combining all operations."""

    def __init__(self):
        self.geometric = GeometricProcessor()
        self.blend_processor = TiledBlendProcessor()
        self.color_adjustments = PhotoshopColorAdjustments()

    def process_single_artwork(self, artwork_data, transform_points, output_size, 
                            template_name, smart_object_name, aspect_ratio, 
                            record_id, white_backing_data=None, 
                            background_data=None, blend_mode='multiply', 
                            opacity=255, manual_adjustments_json=None,
                            tile_size=(128, 128)):
        """
        Complete pipeline: load → transform → blend → adjust → save.

        Args:
            artwork_data: BytesIO object with artwork image data
            transform_points: List of (x,y) coordinates defining frame placement
            output_size: Tuple of (width, height) for output
            template_name: Name of the template
            smart_object_name: Name of the smart object layer
            aspect_ratio: Aspect ratio string (e.g., '1:1')
            record_id: Airtable record ID
            white_backing_data: BytesIO object with white backing image
            background_data: BytesIO object with background (fallback)
            blend_mode: Primary blend mode ('multiply', 'luminosity', etc.)
            opacity: Opacity value (0-255)
            manual_adjustments_json: JSON string with Photoshop-style adjustments
            tile_size: Tuple for tile dimensions (default 128x128)
            
        Returns:
            Tuple of (s3_url, adjustments_applied)
        """
        # Variables for cleanup tracking
        # Variables for cleanup tracking
        cleanup_vars = {
            'artwork': None,
            'white_backing': None,
            'artwork_resized': None,
            'warped_artwork': None,
            'multiply_result': None,  # NEW: Track multiply result
            'blended_result': None,
            'final_result': None,
            'output_buffer': None,
            'warped_buffer': None,
            'backing_buffer': None
        }

        try:
            logger.info(f"Starting complete artwork processing for {smart_object_name}")
            
            # Validate inputs
            if not white_backing_data:
                raise ValueError("White backing data is required for multiply blend mode")
            
            # Step 1: Load and prepare images
            cleanup_vars['artwork'] = Image.open(artwork_data).convert("RGBA")
            cleanup_vars['white_backing'] = Image.open(white_backing_data).convert("RGBA")
            
            logger.info(f"Artwork: {cleanup_vars['artwork'].size}, White backing: {cleanup_vars['white_backing'].size}")
            
            # Step 2: Calculate frame region and target dimensions
            frame_region = self.geometric.calculate_frame_region_with_padding(transform_points, padding=4)
            target_width, target_height = self.geometric.calculate_target_dimensions(transform_points)
            
            logger.info(f"Target artwork dimensions: {target_width}x{target_height}")
            
            # Step 3: Resize artwork
            from PIL import ImageOps
            cleanup_vars['artwork_resized'] = ImageOps.contain(
                cleanup_vars['artwork'], 
                (target_width, target_height)
            )
            
            # Free original artwork
            del cleanup_vars['artwork']
            cleanup_vars['artwork'] = None
            gc.collect()
            
            # Step 4: Apply perspective transformation
            source_points = [
                (0, 0),
                (cleanup_vars['artwork_resized'].width, 0),
                (cleanup_vars['artwork_resized'].width, cleanup_vars['artwork_resized'].height),
                (0, cleanup_vars['artwork_resized'].height)
            ]
            
            logger.info("Applying perspective transform...")
            cleanup_vars['warped_artwork'] = self.geometric.apply_perspective_transform(
                cleanup_vars['artwork_resized'],
                source_points,
                transform_points,
                output_size
            )
            
            # Apply edge smoothing
            cleanup_vars['warped_artwork'] = self.geometric.smooth_edges(cleanup_vars['warped_artwork'])
            
            # Free resized artwork
            del cleanup_vars['artwork_resized']
            cleanup_vars['artwork_resized'] = None
            gc.collect()
            
            # Step 5: Upload images for tiled blending
            sanitized_template_name = template_name.replace(' ', '_').lower()
            sanitized_aspect_ratio = aspect_ratio.replace(':', '_')
            
            # Upload warped artwork
            cleanup_vars['warped_buffer'] = BytesIO()
            cleanup_vars['warped_artwork'].save(cleanup_vars['warped_buffer'], format='PNG')
            cleanup_vars['warped_buffer'].seek(0)
            
            warped_key = f"temp-warped/{sanitized_template_name}_{smart_object_name}_{sanitized_aspect_ratio}_{record_id}_temp.png"
            warped_url = s3_handler.upload_file_obj(cleanup_vars['warped_buffer'], warped_key, content_type='image/png')
            
            # Upload white backing
            cleanup_vars['backing_buffer'] = BytesIO()
            cleanup_vars['white_backing'].save(cleanup_vars['backing_buffer'], format='PNG')
            cleanup_vars['backing_buffer'].seek(0)
            
            backing_key = f"temp-backing/{sanitized_template_name}_{aspect_ratio}_{record_id}_temp.png"
            backing_url = s3_handler.upload_file_obj(cleanup_vars['backing_buffer'], backing_key, content_type='image/png')
            
            # Free memory
            del cleanup_vars['warped_artwork'], cleanup_vars['white_backing']
            del cleanup_vars['warped_buffer'], cleanup_vars['backing_buffer']
            cleanup_vars['warped_artwork'] = cleanup_vars['white_backing'] = None
            cleanup_vars['warped_buffer'] = cleanup_vars['backing_buffer'] = None
            gc.collect()
            
            # Step 6A: Apply vectorized tiled blending
            opacity_normalized = opacity / 255.0

            # Get global luminosity strength from manual adjustments JSON (0-100 scale, default to 20)
            luminosity_strength = 20  # Default
            if manual_adjustments_json:
                global_settings = self.color_adjustments.parse_global_settings(manual_adjustments_json)
                luminosity_strength = global_settings.get('luminosity_blend_strength', 20)

            # Convert to 0-1 opacity for blending
            luminosity_opacity = luminosity_strength / 100.0

            logger.info(f"Applying {blend_mode} blend mode + luminosity blend (strength: {luminosity_strength}%) with vectorized tiled processing")

            # First pass: Apply multiply blend mode
            cleanup_vars['multiply_result'] = self.blend_processor.blend_images_tiled(
                warped_url,
                backing_url,
                frame_region,
                blend_mode=blend_mode,  # Usually 'multiply'
                tile_size=tile_size,
                opacity=opacity_normalized
            )

            logger.info(f"Completed multiply blend, now applying luminosity blend on top")

            # Upload multiply result for luminosity blend input
            multiply_buffer = BytesIO()
            cleanup_vars['multiply_result'].save(multiply_buffer, format='PNG')
            multiply_buffer.seek(0)

            multiply_key = f"temp-multiply/{sanitized_template_name}_{smart_object_name}_{sanitized_aspect_ratio}_{record_id}_multiply.png"
            multiply_url = s3_handler.upload_file_obj(multiply_buffer, multiply_key, content_type='image/png')

            # Free multiply result from memory temporarily
            del cleanup_vars['multiply_result']
            cleanup_vars['multiply_result'] = None
            del multiply_buffer
            gc.collect()

            # Second pass: Apply luminosity blend mode on top
            cleanup_vars['blended_result'] = self.blend_processor.blend_images_tiled(
                warped_url,  # Original artwork (luminosity source)
                multiply_url,  # Multiply result (base)
                frame_region,
                blend_mode='luminosity',
                tile_size=tile_size,
                opacity=luminosity_opacity  # Configurable luminosity strength (default 20%)
            )

            logger.info(f"Completed luminosity blend ({luminosity_opacity:.1%} opacity) on top of multiply result")


            
            # Step 7: Apply Photoshop-style color adjustments
            manual_adjustments = None
            if manual_adjustments_json:
                manual_adjustments = self.color_adjustments.parse_adjustments(manual_adjustments_json, smart_object_name)
            
            if manual_adjustments:
                logger.info(f"Applying Photoshop-style color adjustments for {smart_object_name}")
                cleanup_vars['final_result'] = self.color_adjustments.apply_adjustments_vectorized(
                    cleanup_vars['blended_result'], 
                    manual_adjustments
                )
                
                # Free blended result
                del cleanup_vars['blended_result']
                cleanup_vars['blended_result'] = None
                gc.collect()
            else:
                logger.info("No color adjustments to apply")
                cleanup_vars['final_result'] = cleanup_vars['blended_result']
                cleanup_vars['blended_result'] = None
            
            # Step 8: Save final result
            cleanup_vars['output_buffer'] = BytesIO()
            cleanup_vars['final_result'].save(
                cleanup_vars['output_buffer'],
                format='PNG',
                optimize=True,
                compress_level=6
            )
            cleanup_vars['output_buffer'].seek(0)

            # Calculate file size
            file_size_mb = len(cleanup_vars['output_buffer'].getvalue()) / (1024 * 1024)
            
            # Upload final result
            output_key = f"warped-artworks/{sanitized_template_name}_{smart_object_name}_{sanitized_aspect_ratio}_{record_id}_final.png"
            s3_url = s3_handler.upload_file_obj(cleanup_vars['output_buffer'], output_key, content_type='image/png')
            
            
            logger.info(f"Successfully created final artwork: {s3_url} ({file_size_mb:.2f}MB)")
            
            # Return results
            adjustments_applied = {
                'blend_mode': blend_mode,
                'opacity': opacity_normalized,
                'tile_size': tile_size,
                'frame_region': frame_region,
                'color_adjustments_applied': manual_adjustments is not None,
                'color_adjustments': manual_adjustments,
                'file_size_mb': file_size_mb
            }
            
            return s3_url, adjustments_applied  
        except Exception as e:
            logger.error(f"Error in artwork processing: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None
        finally:
            # Comprehensive cleanup
            for var_name, var_obj in cleanup_vars.items():
                try:
                    if var_obj is not None:
                        del var_obj
                except:
                    pass
            gc.collect()

# ==========================================
# MAIN MOCKUP PROCESSING FUNCTION
# ==========================================
def process_mockup(template_name, backgrounds, coordinates, blend_modes, 
                 artworks, record_id, use_local_mockup=True, 
                 intelligent_scaling=True, manual_adjustments_json=None,
                 white_backings=None):
    """
    Process mockup with multiply blend modes and Photoshop-style color adjustments.

    Args:
    template_name: Name of the template
    backgrounds: Dictionary of aspect ratios to background URLs
    coordinates: Dictionary of aspect ratios to coordinates JSON
    blend_modes: Dictionary of aspect ratios to blend modes JSON
    artworks: Dictionary of layer names to artwork URLs
    record_id: Airtable record ID
    use_local_mockup: Whether to use local mockup generation
    intelligent_scaling: Whether to use intelligent scaling
    manual_adjustments_json: JSON string with Photoshop-style adjustments
    white_backings: Dictionary of aspect ratios to white backing URLs

    Returns:
    Dictionary with results including adjustment JSON
    """
    # Initialize processor
    artwork_processor = ArtworkProcessor()

    # Track whether we're using manual adjustments
    using_manual_adjustments = manual_adjustments_json and manual_adjustments_json.strip() != ""

    logger.info(f"Processing mockup with blend modes and color adjustments")
    logger.info(f"Using manual adjustments: {using_manual_adjustments}")
    logger.info(f"White backings provided: {white_backings is not None}")

    results = {
        'warped_artworks': {},
        'mockup_urls': {},
        'aspect_ratios_processed': [],
        'adjustment_settings': {}
    }

    # Filter artworks
    valid_artworks = {k: v for k, v in artworks.items() if v}
    logger.info(f"Valid artworks: {list(valid_artworks.keys())}")

    # Generate default adjustment JSON if no manual adjustments provided
    if not using_manual_adjustments:
        logger.info("Generating default adjustment settings")
        default_adjustments_json = PhotoshopColorAdjustments.create_adjustments_json(
        list(valid_artworks.keys())
        )
        results['default_adjustments_json'] = default_adjustments_json

    for aspect_ratio, background_url in backgrounds.items():
        if not background_url:
            logger.info(f"No background provided for aspect ratio {aspect_ratio}, skipping")
            continue

        # Check if we have coordinates for this aspect ratio
        if aspect_ratio not in coordinates or not coordinates[aspect_ratio]:
            logger.warning(f"No coordinates for aspect ratio {aspect_ratio}, skipping")
            continue

        try:
            # Parse coordinates JSON
            try:
                full_data = json.loads(coordinates[aspect_ratio])
                coord_data = full_data.get('coordinates', {})
                aspect_blend_modes = full_data.get('blend_modes', {})
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in coordinates for {aspect_ratio}")
                continue

            # Override with separate blend modes if provided
            if aspect_ratio in blend_modes:
                blend_data = blend_modes[aspect_ratio]
                if isinstance(blend_data, str):
                    try:
                        blend_data = json.loads(blend_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse blend modes for {aspect_ratio}")
                        blend_data = {}

                if 'blend_modes' in blend_data:
                    aspect_blend_modes = blend_data['blend_modes']
                else:
                    aspect_blend_modes = blend_data

            # Download background image for dimension checking
            background_data = s3_handler.download_file(background_url)

            # Get white backing if available
            white_backing_data = None
            if white_backings and aspect_ratio in white_backings and white_backings[aspect_ratio]:
                try:
                    white_backing_url = white_backings[aspect_ratio]
                    white_backing_data = s3_handler.download_file(white_backing_url)
                    logger.info(f"Downloaded white backing for {aspect_ratio}: {white_backing_url}")
                except Exception as wb_error:
                    logger.warning(f"Could not download white backing for {aspect_ratio}: {str(wb_error)}")
                    white_backing_data = None
            else:
                logger.info(f"No white backing provided for {aspect_ratio}")

            # Load background and check size for scaling
            bg_image = Image.open(background_data)
            width, height = bg_image.width, bg_image.height
            orig_width, orig_height = width, height

            # Check if background needs downscaling for processing
            total_pixels = width * height
            MAX_PROCESSING_PIXELS = 9000000  # ~9 megapixels

            if total_pixels > MAX_PROCESSING_PIXELS:
                scale_factor = (MAX_PROCESSING_PIXELS / total_pixels) ** 0.5
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                logger.info(f"Background is large ({width}x{height}), downscaling to {new_width}x{new_height}")

                # Scale down the background
                bg_image = bg_image.resize((new_width, new_height), Image.LANCZOS)

                # Scale down white backing if available
                if white_backing_data:
                    try:
                        wb_image = Image.open(white_backing_data)
                        wb_image_scaled = wb_image.resize((new_width, new_height), Image.LANCZOS)

                        wb_buffer = BytesIO()
                        wb_image_scaled.save(wb_buffer, format='PNG')
                        wb_buffer.seek(0)
                        white_backing_data = wb_buffer

                        del wb_image, wb_image_scaled
                        gc.collect()
                    except Exception as wb_scale_error:
                        logger.warning(f"Could not scale white backing: {str(wb_scale_error)}")

                # Scale coordinates
                scaled_coord_data = {}
                for obj_name, points in coord_data.items():
                    scaled_points = []
                    for point in points:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            scaled_points.append((point[0] * scale_factor, point[1] * scale_factor))
                        else:
                            logger.warning(f"Invalid point format in {obj_name}: {point}")
                            scaled_points = []
                            break

                    if scaled_points:
                        scaled_coord_data[obj_name] = scaled_points

                coord_data = scaled_coord_data

                # Update background data
                bg_buffer = BytesIO()
                bg_image.save(bg_buffer, format='PNG')
                bg_buffer.seek(0)
                background_data = bg_buffer

            output_size = (bg_image.width, bg_image.height)
            background_data.seek(0)
            if white_backing_data:
                white_backing_data.seek(0)

            # Filter coordinates to available artworks
            available_coord_data = {}
            for smart_obj_name, artwork_url in valid_artworks.items():
                coord_key = None
                # Try multiple naming variations
                if smart_obj_name in coord_data:
                    coord_key = smart_obj_name
                elif smart_obj_name.startswith('Print') and len(smart_obj_name) > 5:
                    spaced_name = smart_obj_name[:5] + ' ' + smart_obj_name[5:]
                    if spaced_name in coord_data:
                        coord_key = spaced_name
                else:
                    no_space_name = smart_obj_name.replace(' ', '')
                    if no_space_name in coord_data:
                        coord_key = no_space_name

                if coord_key:
                    available_coord_data[smart_obj_name] = coord_data[coord_key]

            logger.info(f"Available coordinates for {len(available_coord_data)} artworks: {list(available_coord_data.keys())}")

            # Process each artwork with complete pipeline
            warped_artworks = {}
            for smart_obj_name, artwork_url in valid_artworks.items():
                if smart_obj_name not in available_coord_data:
                    logger.warning(f"No coordinates for {smart_obj_name} in {aspect_ratio}, skipping")
                    continue

                # Get transform points
                transform_points = available_coord_data[smart_obj_name]

                # Get blend info
                blend_info = {}
                for potential_name in [smart_obj_name, smart_obj_name.replace(' ', ''), smart_obj_name.replace('Print', 'Print ')]:
                    if potential_name in aspect_blend_modes:
                        blend_info = aspect_blend_modes[potential_name]
                        break

                blend_mode = blend_info.get('blend_mode', 'multiply')  # Default to multiply
                opacity = blend_info.get('opacity', 255)

                # Download artwork
                artwork_data = s3_handler.download_file(artwork_url)

                # Prepare data copies
                background_copy = BytesIO(background_data.getvalue())
                white_backing_copy = None
                if white_backing_data:
                    white_backing_copy = BytesIO(white_backing_data.getvalue())

                try:
                    # Use the complete artwork processing pipeline
                    if white_backing_copy:
                        logger.info(f"Processing {smart_obj_name} with complete pipeline")

                        # NEW: Pass blend_info to the processor
                        # artwork_processor._current_blend_info = blend_info  # Store blend info temporarily

                        result = artwork_processor.process_single_artwork(
                            artwork_data,
                            transform_points,
                            output_size,
                            template_name,
                            smart_obj_name,
                            aspect_ratio,
                            record_id,
                            white_backing_copy,
                            background_copy,
                            blend_mode,
                            opacity,
                            manual_adjustments_json
                        )

                        # Clean up temporary blend info
                        # artwork_processor._current_blend_info = None
                    else:
                        # Fallback for no white backing
                        logger.warning(f"No white backing for {smart_obj_name}, skipping")
                        continue

                    # Handle result
                    if isinstance(result, tuple) and len(result) == 2:
                        warped_url, adjustment_info = result
                        if warped_url:
                            warped_artworks[smart_obj_name] = warped_url
                            # Store adjustment info for this artwork
                            if adjustment_info and 'color_adjustments' in adjustment_info:
                                results['adjustment_settings'][smart_obj_name] = adjustment_info['color_adjustments']
                            logger.info(f"Successfully processed {smart_obj_name}: {warped_url}")
                    else:
                        warped_url = result
                        if warped_url:
                            warped_artworks[smart_obj_name] = warped_url
                            logger.info(f"Successfully processed {smart_obj_name}: {warped_url}")

                except Exception as e:
                    logger.error(f"Error processing {smart_obj_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

                # Clean up
                del artwork_data, background_copy
                if white_backing_copy:
                    del white_backing_copy
                gc.collect()

            # Clean up background resources
            del background_data, bg_image
            if white_backing_data:
                del white_backing_data
            gc.collect()

            # Create final mockup using local generator
            if warped_artworks:
                try:
                    if use_local_mockup:
                        from local_mockup_generator import create_local_mockup, extract_blend_info

                        blend_modes_dict, opacities_dict = extract_blend_info(coordinates[aspect_ratio])

                        logger.info(f"Creating local mockup for {aspect_ratio} with {len(warped_artworks)} layers")

                        bg_original_dimensions = (orig_width, orig_height)

                        mockup_url = create_local_mockup(
                            background_url,
                            warped_artworks,
                            blend_modes=blend_modes_dict,
                            opacities=opacities_dict,
                            chunk_size=500,
                            record_id=record_id,
                            intelligent_scaling=intelligent_scaling,
                            original_dimensions=bg_original_dimensions
                        )

                        if mockup_url:
                            results['mockup_urls'][aspect_ratio] = mockup_url
                            results['aspect_ratios_processed'].append(aspect_ratio)
                            logger.info(f"Created local mockup for {aspect_ratio}: {mockup_url}")
                    else:
                        # Use Renderform as fallback
                        mockup_result = renderform_handler.create_mockup(
                            aspect_ratio,
                            background_url,
                            warped_artworks
                        )
                        if mockup_result and 'href' in mockup_result:
                            results['mockup_urls'][aspect_ratio] = mockup_result['href']
                            results['aspect_ratios_processed'].append(aspect_ratio)
                            logger.info(f"Created Renderform mockup for {aspect_ratio}: {mockup_result['href']}")
                except Exception as e:
                    logger.error(f"Error creating mockup for {aspect_ratio}: {str(e)}")

            # Store warped artwork URLs
            if aspect_ratio not in results['warped_artworks']:
                results['warped_artworks'][aspect_ratio] = {}
            results['warped_artworks'][aspect_ratio].update(warped_artworks)

            logger.info(f"Successfully processed mockup for {aspect_ratio}")

        except Exception as e:
            logger.error(f"Error processing mockup for aspect ratio {aspect_ratio}: {str(e)}")
            logger.error(traceback.format_exc())

    # Format response for Airtable
    airtable_updates = format_airtable_response(results, using_manual_adjustments)

    return {
    'airtable_updates': airtable_updates,
    'detailed_results': results
    }

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def format_airtable_response(results, using_manual_adjustments):
   """Format processing results for Airtable update."""
   airtable_updates = {}
   
   # Add warped artwork URLs
   for aspect_ratio, artworks_dict in results['warped_artworks'].items():
       for print_name, url in artworks_dict.items():
           field_name = f"{print_name} ({aspect_ratio})"
           airtable_updates[field_name] = url
   
   # Add mockup URLs
   for aspect_ratio, url in results['mockup_urls'].items():
       airtable_updates[f"Ad Mockup ({aspect_ratio})"] = url
   
   # Handle adjustment JSON - either save default or preserve existing manual adjustments
   if not using_manual_adjustments and 'default_adjustments_json' in results:
       # First run - save default adjustment settings for user to edit
       airtable_updates['Mockup Adjustments'] = results['default_adjustments_json']
       logger.info("Saving default adjustment settings to Airtable for first-time setup")
   elif using_manual_adjustments:
       # Using manual adjustments - don't overwrite the field
       logger.info("Using manual adjustments - preserving existing Mockup Adjustments field")
   else:
       # No adjustments to save
       logger.info("No adjustment settings to update")
   
   # Update status
   if results['aspect_ratios_processed']:
       if not using_manual_adjustments:
           process_status = f"Success: Processed {len(results['aspect_ratios_processed'])} mockups with default settings (edit Mockup Adjustments to fine-tune colors)"
       else:
           num_adjusted = len([k for k, v in results['adjustment_settings'].items() if v])
           process_status = f"Success: Processed {len(results['aspect_ratios_processed'])} mockups with {num_adjusted} custom color adjustments"
   else:
       process_status = "Failed: No mockups could be processed"
   
   airtable_updates['Upload Status'] = process_status
   
   return airtable_updates

def get_memory_usage_mb():
   """Get current memory usage in MB, with fallback if psutil is not available"""
   try:
       import psutil
       import os
       process = psutil.Process(os.getpid())
       return process.memory_info().rss / (1024 * 1024)
   except ImportError:
       return -1  # Indicates memory usage couldn't be determined

def extract_surrounding_region(background_image, transform_points, expansion_factor=0.5):
   """
   Extract a region surrounding (but excluding) the artwork placement area.
   Kept for fallback compatibility.
   """
   try:
       # Ensure transform_points are in the right format (tuples or lists)
       points = []
       for point in transform_points:
           if isinstance(point, (list, tuple)) and len(point) >= 2:
               points.append((float(point[0]), float(point[1])))
           else:
               raise ValueError(f"Invalid point format: {point}")
               
       # Calculate the bounding box of the transform points
       x_coords = [p[0] for p in points]
       y_coords = [p[1] for p in points]
       min_x, max_x = min(x_coords), max(x_coords)
       min_y, max_y = min(y_coords), max(y_coords)
       
       # Calculate center point
       center_x = (min_x + max_x) / 2
       center_y = (min_y + max_y) / 2
       
       # Calculate width and height of the original bounding box
       width = max_x - min_x
       height = max_y - min_y
       
       # Calculate expanded bounding box (with expansion_factor)
       expanded_width = width * (1 + expansion_factor * 2)  # Expand on both sides
       expanded_height = height * (1 + expansion_factor * 2)
       
       # Calculate expanded bounds, ensuring they stay within image dimensions
       exp_min_x = max(0, int(center_x - expanded_width / 2))
       exp_max_x = min(background_image.width, int(center_x + expanded_width / 2))
       exp_min_y = max(0, int(center_y - expanded_height / 2))
       exp_max_y = min(background_image.height, int(center_y + expanded_height / 2))
       
       # Create a mask for the expanded region
       mask = Image.new('L', background_image.size, 0)
       mask_draw = ImageDraw.Draw(mask)
       
       # Fill the expanded rectangle
       mask_draw.rectangle([(exp_min_x, exp_min_y), (exp_max_x, exp_max_y)], fill=255)
       
       # Now subtract the original artwork area by filling it with black
       mask_draw.polygon(points, fill=0)
       
       # Use the mask to extract the region
       surrounding_region = Image.new('RGBA', background_image.size, (0, 0, 0, 0))
       surrounding_region.paste(background_image, (0, 0), mask)
       
       # Crop to the expanded bounds
       cropped_region = surrounding_region.crop((exp_min_x, exp_min_y, exp_max_x, exp_max_y))
       
       # Safety check for empty images
       if cropped_region.size[0] <= 1 or cropped_region.size[1] <= 1 or cropped_region.getextrema()[0][1] == 0:
           logger.warning("Extracted surrounding region is too small or empty, using full background")
           # Use a portion of the background instead
           full_width, full_height = background_image.size
           return background_image.crop((0, 0, min(full_width, 800), min(full_height, 800)))
           
       return cropped_region
   except Exception as e:
       logger.error(f"Error extracting surrounding region: {str(e)}")
       logger.error(traceback.format_exc())
       # Return a portion of the background as fallback
       return background_image

def analyze_image_colors(image):
   """
   Analyze key color statistics of an image.
   Kept for compatibility and diagnostics.
   """
   try:
       from PIL import ImageStat
       
       # Convert image to RGBA to handle transparency
       img_rgba = image.convert('RGBA')

       # Split into channels
       r, g, b, a = img_rgba.split()

       # Threshold alpha to create a mask (1 where alpha >= 10, else 0)
       alpha_threshold = 10
       alpha_mask = a.point(lambda p: 255 if p >= alpha_threshold else 0)

       # Merge RGB back to RGB image (drops alpha)
       rgb_image = Image.merge("RGB", (r, g, b))

       # Use ImageStat with the alpha mask to compute stats only on non-transparent pixels
       stats = ImageStat.Stat(rgb_image, mask=alpha_mask)
       
       # Get mean values per channel
       r_mean, g_mean, b_mean = stats.mean
       
       # Calculate brightness (0-255)
       brightness = sum(stats.mean) / 3
       
       # Calculate color variance as a measure of colorfulness
       r_var, g_var, b_var = stats.var
       colorfulness = sum([r_var, g_var, b_var]) / 3
       
       # Calculate contrast
       extrema = rgb_image.getextrema()
       r_min, r_max = extrema[0]
       g_min, g_max = extrema[1]
       b_min, b_max = extrema[2]
       contrast = (r_max - r_min + g_max - g_min + b_max - b_min) / 3
       
       # Calculate saturation (0-1 range)
       max_mean = max(r_mean, g_mean, b_mean)
       min_mean = min(r_mean, g_mean, b_mean)
       
       saturation = 0
       if max_mean > 0:
           saturation = (max_mean - min_mean) / max_mean
       
       # Calculate color temperature (warm vs cool)
       color_temp = r_mean / (b_mean if b_mean > 0.1 else 0.1)
       
       # Calculate dominant color tone
       if r_mean > g_mean and r_mean > b_mean:
           dominant_tone = "red"
       elif g_mean > r_mean and g_mean > b_mean:
           dominant_tone = "green"
       else:
           dominant_tone = "blue"
       
       # Detect if image is mostly monochrome
       color_variance = abs(r_mean - g_mean) + abs(r_mean - b_mean) + abs(g_mean - b_mean)
       is_monochrome = color_variance < 30  # Threshold for considering image monochromatic
       
       return {
           'brightness': brightness,
           'saturation': saturation,
           'contrast': contrast,
           'colorfulness': colorfulness,
           'color_temp': color_temp,
           'r_mean': r_mean,
           'g_mean': g_mean,
           'b_mean': b_mean,
           'dominant_tone': dominant_tone,
           'is_monochrome': is_monochrome
       }
   except Exception as e:
       logger.error(f"Error analyzing image colors: {str(e)}")
       # Return default values as fallback
       return {
           'brightness': 128,
           'saturation': 0.5,
           'contrast': 128,
           'colorfulness': 5000,
           'color_temp': 1.0,
           'r_mean': 128,
           'g_mean': 128,
           'b_mean': 128,
           'dominant_tone': "neutral",
           'is_monochrome': False
       }

# ==========================================
# FALLBACK FUNCTIONS (for compatibility)
# ==========================================

def warp_artwork(artwork_data, transform_points, output_size, template_name, 
                smart_object_name, aspect_ratio, record_id, background_data=None, 
                blend_mode='normal', opacity=255, manual_adjustments=None):
   """
   Legacy fallback function for backward compatibility.
   Redirects to the new artwork processor.
   """
   logger.warning("Using legacy warp_artwork function - consider migrating to ArtworkProcessor")
   
   # Create processor and use the new pipeline
   processor = ArtworkProcessor()
   
   # Convert manual adjustments to JSON format if needed
   manual_adjustments_json = None
   if manual_adjustments:
       # Wrap single adjustment in JSON structure
       temp_json = {smart_object_name: manual_adjustments}
       manual_adjustments_json = json.dumps(temp_json)
   
   return processor.process_single_artwork(
       artwork_data, transform_points, output_size, template_name,
       smart_object_name, aspect_ratio, record_id, 
       white_backing_data=None,  # No white backing in legacy mode
       background_data=background_data,
       blend_mode=blend_mode,
       opacity=opacity,
       manual_adjustments_json=manual_adjustments_json
   )

# ==========================================
# TESTING AND DEBUGGING FUNCTIONS
# ==========================================

def test_photoshop_adjustments():
   """
   Test function to verify Photoshop-style adjustments work correctly.
   """
   color_adjustments = PhotoshopColorAdjustments()
   
   # Create test adjustment JSON
   test_adjustments = {
       "Print1": {
           "brightness": 20,      # +20% brighter
           "contrast": 15,        # +15% more contrast
           "saturation": -10,     # -10% less saturated
           "warmth": 25,          # +25% warmer
           "shadows": 30,         # +30% brighter shadows
           "clarity": 10          # +10% more clarity
       },
       "Print2": {
           "brightness": -5,      # -5% darker
           "vibrance": 20,        # +20% more vibrant
           "highlights": -15,     # -15% darker highlights
           "tint": 5              # +5% more magenta
       }
   }
   
   # Generate JSON
   smart_objects = ["Print1", "Print2", "Print3"]
   adjustments_json = color_adjustments.create_adjustments_json(smart_objects, test_adjustments.get("Print1"))
   
   logger.info("Generated test adjustments JSON:")
   logger.info(adjustments_json)
   
   # Test parsing
   parsed = color_adjustments.parse_adjustments(adjustments_json, "Print1")
   logger.info(f"Parsed adjustments for Print1: {parsed}")
   
   return adjustments_json

def test_blend_modes():
   """Test function to verify blend modes work correctly."""
   blend_modes = PhotoshopBlendModes()
   
   # Create test images
   test_size = (100, 100)
   
   # Red image
   red_array = np.full((test_size[1], test_size[0], 4), [255, 0, 0, 255], dtype=np.uint8)
   
   # Blue image  
   blue_array = np.full((test_size[1], test_size[0], 4), [0, 0, 255, 255], dtype=np.uint8)
   
   # Test multiply blend
   multiply_result = blend_modes.multiply_blend(red_array, blue_array)
   logger.info(f"Multiply blend result shape: {multiply_result.shape}")
   logger.info(f"Multiply blend sample pixel: {multiply_result[50, 50]}")
   
   # Test luminosity blend
   luminosity_result = blend_modes.luminosity_blend(blue_array, red_array, 0.5)
   logger.info(f"Luminosity blend result shape: {luminosity_result.shape}")
   logger.info(f"Luminosity blend sample pixel: {luminosity_result[50, 50]}")
   
   return True

if __name__ == "__main__":
   # Run tests if this file is executed directly
   logger.info("Running processor.py tests...")
   #test_photoshop_adjustments()
   #test_blend_modes()
   logger.info("Tests completed!")
