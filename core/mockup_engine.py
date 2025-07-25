"""
Mockup Engine - Professional automated mockup generation

This module provides the core processing pipeline for generating mockups with:
- Photoshop-style blend modes (multiply, luminosity)
- Perspective transformation and warping
- Memory-efficient tiled processing
- Intelligent color adjustments

Designed to work with Photoshop-prepped assets for lightweight, scalable processing - runs on as little as 512MB RAM.
"""

import logging
import traceback
from io import BytesIO
from PIL import Image, ImageEnhance, ImageDraw, ImageStat, ImageOps, ImageChops  
import cv2
import numpy as np
import json
import time
import gc

logger = logging.getLogger(__name__)

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
    
    def blend_images_tiled(self, top_image, bottom_image, frame_region, 
                        blend_mode='multiply', tile_size=(128, 128), 
                        opacity=1.0, padding=4):
        """
        Memory-efficient tiled blending with automatic masking.
        
        Args:
            top_image: PIL Image (top layer)
            bottom_image: PIL Image (bottom layer) 
            frame_region: Tuple (min_x, min_y, max_x, max_y) defining blend area
            blend_mode: Blend mode ('multiply', 'luminosity', etc.)
            tile_size: Tuple (width, height) for tile dimensions
            opacity: Blend opacity (0.0-1.0)
            padding: Padding around frame region
            
        Returns:
            PIL Image with blended result
        """
        # Variables for cleanup
        result_image = None
        artwork_mask = None
        
        try:
            # Ensure images are RGBA
            if top_image.mode != 'RGBA':
                top_image = top_image.convert('RGBA')
            if bottom_image.mode != 'RGBA':
                bottom_image = bottom_image.convert('RGBA')
            
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
            
            # Make mask more precise by using artwork's alpha channel
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
            # Cleanup
            if artwork_mask:
                del artwork_mask
            gc.collect()

# ==========================================
# PHOTOSHOP-STYLE COLOR ADJUSTMENTS
# ==========================================

class PhotoshopColorAdjustments:
    """Handles Photoshop-style color adjustments with -100 to +100 scale."""
    
    @staticmethod
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
                "luminosity_blend_strength": 20,  # Global luminosity blend (0-100)
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
            adjustment_type: Type of adjustment
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
               'brightness', 'contrast', 'gamma', 'clarity', 'structure',
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
# MAIN MOCKUP ENGINE
# ==========================================

class MockupEngine:
   """
   Main mockup processing engine that combines all components.
   
   This is the primary interface for generating mockups with:
   - Perspective transformation
   - Photoshop-style blend modes
   - Memory-efficient processing
   - Color adjustments
   """

   def __init__(self):
       self.geometric = GeometricProcessor()
       self.blend_processor = TiledBlendProcessor()
       self.color_adjustments = PhotoshopColorAdjustments()

   def process_mockup(self, artwork, background, color_backing, placement_coords,
                     blend_mode='multiply', opacity=0.8, adjustments=None,
                     tile_size=(128, 128), output_path=None):
       """
       Complete mockup processing pipeline.

       Args:
           artwork: PIL Image - The artwork to place
           background: PIL Image - Background template with transparent artwork area
           color_backing: PIL Image - Color-corrected backing from Photoshop
           placement_coords: List of (x,y) tuples defining where artwork goes
           blend_mode: Primary blend mode ('multiply', 'luminosity', 'normal')
           opacity: Blend opacity (0.0-1.0)
           adjustments: Dict with Photoshop-style adjustments (-100 to +100)
           tile_size: Tuple for memory-efficient processing
           output_path: Optional path to save result
           
       Returns:
           PIL Image with final mockup
       """
       cleanup_vars = {
           'artwork_resized': None,
           'warped_artwork': None,
           'multiply_result': None,
           'luminosity_result': None,
           'final_result': None
       }

       try:
           logger.info(f"Starting mockup processing with {blend_mode} blend mode")
           
           # Ensure all images are RGBA
           artwork = artwork.convert('RGBA') if artwork.mode != 'RGBA' else artwork
           background = background.convert('RGBA') if background.mode != 'RGBA' else background
           color_backing = color_backing.convert('RGBA') if color_backing.mode != 'RGBA' else color_backing
           
           # Step 1: Calculate target dimensions for artwork
           target_width, target_height = self.geometric.calculate_target_dimensions(placement_coords)
           output_size = background.size
           
           logger.info(f"Target artwork size: {target_width}x{target_height}")
           logger.info(f"Output size: {output_size}")
           
           # Step 2: Resize artwork to fit target dimensions
           cleanup_vars['artwork_resized'] = ImageOps.contain(artwork, (target_width, target_height))
           
           # Step 3: Apply perspective transformation
           source_points = [
               (0, 0),
               (cleanup_vars['artwork_resized'].width, 0),
               (cleanup_vars['artwork_resized'].width, cleanup_vars['artwork_resized'].height),
               (0, cleanup_vars['artwork_resized'].height)
           ]
           
           logger.info("Applying perspective transformation...")
           cleanup_vars['warped_artwork'] = self.geometric.apply_perspective_transform(
               cleanup_vars['artwork_resized'],
               source_points,
               placement_coords,
               output_size
           )
           
           # Apply edge smoothing
           cleanup_vars['warped_artwork'] = self.geometric.smooth_edges(cleanup_vars['warped_artwork'])
           
           # Free resized artwork
           del cleanup_vars['artwork_resized']
           cleanup_vars['artwork_resized'] = None
           gc.collect()
           
           # Step 4: Calculate frame region for tiled processing
           frame_region = self.geometric.calculate_frame_region_with_padding(placement_coords, padding=4)
           
           # Step 5: Apply multiply blend mode
           logger.info(f"Applying {blend_mode} blend with opacity {opacity:.2f}")
           cleanup_vars['multiply_result'] = self.blend_processor.blend_images_tiled(
               cleanup_vars['warped_artwork'],
               color_backing,
               frame_region,
               blend_mode=blend_mode,
               tile_size=tile_size,
               opacity=opacity
           )
           
           # Step 6: Apply luminosity blend for color refinement (optional)
           luminosity_strength = 0.2  # Default 20% luminosity blend
           if adjustments and '_global_settings' in adjustments:
               luminosity_strength = adjustments['_global_settings'].get('luminosity_blend_strength', 20) / 100.0
           
           if luminosity_strength > 0:
               logger.info(f"Applying luminosity blend with {luminosity_strength:.1%} strength")
               cleanup_vars['luminosity_result'] = self.blend_processor.blend_images_tiled(
                   cleanup_vars['warped_artwork'],  # Original artwork for luminosity
                   cleanup_vars['multiply_result'],  # Multiply result as base
                   frame_region,
                   blend_mode='luminosity',
                   tile_size=tile_size,
                   opacity=luminosity_strength
               )
               
               # Free multiply result
               del cleanup_vars['multiply_result']
               cleanup_vars['multiply_result'] = None
               gc.collect()
               
               blended_result = cleanup_vars['luminosity_result']
           else:
               blended_result = cleanup_vars['multiply_result']
           
           # Step 7: Apply color adjustments if provided
           if adjustments and any(k not in ['_metadata', '_global_settings'] for k in adjustments.keys()):
               logger.info("Applying Photoshop-style color adjustments")
               # For single artwork, use first non-metadata key
               artwork_adjustments = None
               for key, value in adjustments.items():
                   if key not in ['_metadata', '_global_settings']:
                       artwork_adjustments = value
                       break
               
               if artwork_adjustments:
                   cleanup_vars['final_result'] = self.color_adjustments.apply_adjustments_vectorized(
                       blended_result, 
                       artwork_adjustments
                   )
               else:
                   cleanup_vars['final_result'] = blended_result
           else:
               logger.info("No color adjustments to apply")
               cleanup_vars['final_result'] = blended_result
           
           # Step 8: Composite with background template
           logger.info("Compositing with background template")
           final_mockup = Image.alpha_composite(background, cleanup_vars['final_result'])
           
           # Step 9: Save if output path provided
           if output_path:
               final_mockup.save(output_path, format='PNG', optimize=True)
               logger.info(f"Saved mockup to {output_path}")
           
           logger.info("Mockup processing completed successfully")
           return final_mockup
           
       except Exception as e:
           logger.error(f"Error in mockup processing: {str(e)}")
           logger.error(traceback.format_exc())
           raise
           
       finally:
           # Comprehensive cleanup
           for var_name, var_obj in cleanup_vars.items():
               try:
                   if var_obj is not None:
                       del var_obj
               except:
                   pass
           gc.collect()

   def process_batch(self, artworks, background, color_backing, placement_coords_list,
                    blend_mode='multiply', opacity=0.8, adjustments_list=None,
                    tile_size=(128, 128), output_dir=None):
       """
       Process multiple artworks with the same template.
       
       Args:
           artworks: List of PIL Images
           background: PIL Image - Background template
           color_backing: PIL Image - Color-corrected backing
           placement_coords_list: List of placement coordinates for each artwork
           blend_mode: Blend mode to use
           opacity: Blend opacity
           adjustments_list: Optional list of adjustment dicts for each artwork
           tile_size: Tile size for processing
           output_dir: Optional directory to save results
           
       Returns:
           List of PIL Images with processed mockups
       """
       results = []
       
       for i, artwork in enumerate(artworks):
           try:
               # Get placement coordinates for this artwork
               placement_coords = placement_coords_list[i] if i < len(placement_coords_list) else placement_coords_list[0]
               
               # Get adjustments for this artwork
               adjustments = adjustments_list[i] if adjustments_list and i < len(adjustments_list) else None
               
               # Generate output path if directory provided
               output_path = None
               if output_dir:
                   output_path = f"{output_dir}/mockup_{i+1:03d}.png"
               
               # Process mockup
               mockup = self.process_mockup(
                   artwork=artwork,
                   background=background,
                   color_backing=color_backing,
                   placement_coords=placement_coords,
                   blend_mode=blend_mode,
                   opacity=opacity,
                   adjustments=adjustments,
                   tile_size=tile_size,
                   output_path=output_path
               )
               
               results.append(mockup)
               logger.info(f"Processed artwork {i+1}/{len(artworks)}")
               
           except Exception as e:
               logger.error(f"Error processing artwork {i+1}: {str(e)}")
               results.append(None)
       
       return results

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_memory_usage_mb():
   """Get current memory usage in MB, with fallback if psutil is not available"""
   try:
       import psutil
       import os
       process = psutil.Process(os.getpid())
       return process.memory_info().rss / (1024 * 1024)
   except ImportError:
       return -1  # Indicates memory usage couldn't be determined

def create_default_adjustments(smart_object_names):
   """
   Create default Photoshop-style adjustment JSON for smart objects.
   
   Args:
       smart_object_names: List of smart object names
       
   Returns:
       JSON string with default adjustments
   """
   return PhotoshopColorAdjustments.create_adjustments_json(smart_object_names)

def parse_adjustments_json(json_string):
   """
   Parse adjustments JSON string into dictionary.
   
   Args:
       json_string: JSON string with adjustments
       
   Returns:
       Dictionary with parsed adjustments
   """
   try:
       return json.loads(json_string)
   except json.JSONDecodeError as e:
       logger.error(f"Error parsing adjustments JSON: {e}")
       return {}

# ==========================================
# TESTING AND DEBUGGING
# ==========================================

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

def test_color_adjustments():
   """Test color adjustment functionality."""
   adjustments = PhotoshopColorAdjustments()
   
   # Create test adjustment JSON
   test_objects = ["Print1", "Print2"]
   adjustments_json = adjustments.create_adjustments_json(test_objects)
   
   logger.info("Generated test adjustments JSON:")
   logger.info(adjustments_json)
   
   # Test parsing
   parsed = adjustments.parse_adjustments(adjustments_json, "Print1")
   logger.info(f"Parsed adjustments for Print1: {parsed}")
   
   return adjustments_json

if __name__ == "__main__":
   # Run tests if this file is executed directly
   logger.info("Running mockup_engine.py tests...")
   test_blend_modes()
   test_color_adjustments()
   logger.info("Tests completed")
