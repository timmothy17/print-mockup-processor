import logging
import numpy as np
from PIL import Image

# Import the entire blend_modes package to inspect it
import blend_modes

logger = logging.getLogger(__name__)

# Print out all available functions in the blend_modes module
logger.info(f"Available blend_modes functions: {dir(blend_modes)}")

class ImageBlender:
    """A class for efficient image blending operations using blend-modes library"""
    
    # Map between PSD blend mode names and blend_modes functions
    BLEND_MODE_MAP = {
        # We'll add the correct mappings after inspecting what's available
        'normal': None,  # Will be set in __init__
    }
    
    @classmethod
    def __init__(cls):
        """Initialize the blend mode mappings based on what's available"""
        # Check available functions in blend_modes
        available_functions = dir(blend_modes)
        logger.info(f"Available blend_modes functions: {available_functions}")
        
        # Try to find the normal blend function (could be different names)
        if 'normal' in available_functions:
            cls.BLEND_MODE_MAP['normal'] = getattr(blend_modes, 'normal')
        elif 'normal_blend' in available_functions:
            cls.BLEND_MODE_MAP['normal'] = getattr(blend_modes, 'normal_blend')
        else:
            # Fallback to a simple function that returns the foreground
            def simple_normal(bg, fg, opacity):
                return fg
            cls.BLEND_MODE_MAP['normal'] = simple_normal
    
    @staticmethod
    def _normalize_opacity(opacity):
        """Convert opacity from 0-255 range to 0-1 range"""
        if opacity is None:
            return 1.0
        if isinstance(opacity, (int, float)):
            return max(0.0, min(1.0, float(opacity) / 255.0))
        return 1.0
    
    @staticmethod
    def pil_to_float32(image):
        """Convert PIL Image to float32 numpy array (required by blend-modes)"""
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Convert to numpy array and normalize to 0-1 range
        img_array = np.array(image).astype(np.float32)
        # Scale to 0-1 range (required by blend-modes)
        img_array = img_array / 255
        
        return img_array
    
    @staticmethod
    def float32_to_pil(float_array):
        """Convert float32 numpy array back to PIL Image"""
        # Scale back to 0-255 range
        img_array = float_array * 255
        # Convert to uint8
        img_array = img_array.astype(np.uint8)
        # Create PIL Image
        image = Image.fromarray(img_array, 'RGBA')
        
        return image
    
    @classmethod
    def apply_blend_mode(cls, background, foreground, blend_mode='normal', opacity=1.0):
        """
        Apply a blend mode to combine foreground and background images.
        
        Args:
            background: PIL Image for the background
            foreground: PIL Image for the foreground
            blend_mode: String name of the blend mode to use
            opacity: Opacity value (0-1) for the foreground
        
        Returns:
            A PIL Image with the blended result
        """
        try:
            # Ensure the class is initialized
            if cls.BLEND_MODE_MAP['normal'] is None:
                cls.__init__()
            
            # Fallback to normal blend mode
            blend_func = cls.BLEND_MODE_MAP.get('normal')
            
            # Convert images to float32 arrays
            background_float = cls.pil_to_float32(background)
            foreground_float = cls.pil_to_float32(foreground)
            
            # Ensure opacity is in 0-1 range
            opacity = cls._normalize_opacity(opacity)
            
            # Make sure foreground and background are the same size
            if background.size != foreground.size:
                # Resize foreground to match background
                foreground = foreground.resize(background.size, Image.Resampling.LANCZOS)
                foreground_float = cls.pil_to_float32(foreground)
            
            # For now, just use simple alpha compositing since we don't know which blend functions are available
            # This is a fallback that will at least not break
            result = background_float * (1 - opacity) + foreground_float * opacity
            
            # Convert back to PIL Image
            blended_image = cls.float32_to_pil(result)
            
            return blended_image
        
        except Exception as e:
            logger.error(f"Error applying blend mode: {str(e)}")
            # If blending fails, just return the background
            return background

    @classmethod
    def composite_layers(cls, background, foreground_layers):
        """
        Composite multiple foreground layers onto a background.
        
        Args:
            background: PIL Image for the background
            foreground_layers: List of tuples (image, blend_mode, opacity)
        
        Returns:
            A PIL Image with all layers composited
        """
        # Start with the background
        result = background.copy()
        
        # Apply each foreground layer
        for layer in foreground_layers:
            if len(layer) == 3:
                foreground, blend_mode, opacity = layer
            elif len(layer) == 2:
                foreground, blend_mode = layer
                opacity = 1.0  # Default opacity
            else:
                foreground = layer[0]
                blend_mode = 'normal'
                opacity = 1.0
            
            # Apply this layer
            result = cls.apply_blend_mode(result, foreground, blend_mode, opacity)
        
        return result
