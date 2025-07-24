import logging
import traceback
from io import BytesIO
from PIL import Image, ImageEnhance, ImageDraw, ImageStat, ImageOps
from psd_tools import PSDImage
import cv2
import numpy as np
import json
import time
import gc  # For garbage collection
from s3_handler import S3Handler
from renderform_handler import RenderformHandler

# Initialize handlers
logger = logging.getLogger(__name__)
s3_handler = S3Handler()
renderform_handler = RenderformHandler()

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
        borderValue=[0, 0, 0]  # Black border
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

def extract_surrounding_region(background_image, transform_points, expansion_factor=0.5):
    """
    Extract a region surrounding (but excluding) the artwork placement area.
    
    Args:
        background_image: PIL Image of the background
        transform_points: List of (x,y) tuples defining the placement area
        expansion_factor: How much to expand the bounding box (0.5 = 50% expansion)
        
    Returns:
        PIL Image containing the surrounding region
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
    
    Args:
        image: PIL Image to analyze
    
    Returns:
        Dictionary with color statistics
    """
    try:
        # Convert to RGB for analysis
        img_rgb = image.convert('RGB')
        
        # Use PIL's ImageStat for efficient calculations
        stats = ImageStat.Stat(img_rgb)
        
        # Get mean values per channel
        r_mean, g_mean, b_mean = stats.mean
        
        # Calculate brightness (0-255)
        brightness = sum(stats.mean) / 3
        
        # Calculate color variance as a measure of colorfulness
        r_var, g_var, b_var = stats.var
        colorfulness = sum([r_var, g_var, b_var]) / 3
        
        # Calculate contrast
        extrema = img_rgb.getextrema()
        r_min, r_max = extrema[0]
        g_min, g_max = extrema[1]
        b_min, b_max = extrema[2]
        contrast = (r_max - r_min + g_max - g_min + b_max - b_min) / 3
        
        # Calculate saturation (0-1 range)
        # For each pixel, saturation is (max-min)/max if max>0 else 0
        # We use an approximation based on the mean values
        max_mean = max(r_mean, g_mean, b_mean)
        min_mean = min(r_mean, g_mean, b_mean)
        
        saturation = 0
        if max_mean > 0:
            saturation = (max_mean - min_mean) / max_mean
        
        # Calculate color temperature (warm vs cool)
        # Simple approximation: ratio of red to blue
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

def calculate_adjustments(artwork_stats, target_stats, config=None):
    """
    Calculate adjustment factors based on color statistics with dynamic constraints.
    Improved handling for color-dominant artworks and multiply blend mode.
    
    Args:
        artwork_stats: Color statistics of the artwork
        target_stats: Color statistics of the target region
        config: Optional configuration parameters
    
    Returns:
        Dictionary with adjustment factors
    """
    # Get config or use defaults
    config = config or {}
    
    # Calculate the "difference factor" between the images (0-1 range)
    brightness_diff = abs(artwork_stats['brightness'] - target_stats['brightness']) / 255
    saturation_diff = abs(artwork_stats['saturation'] - target_stats['saturation'])
    color_temp_diff = abs(artwork_stats['color_temp'] - target_stats['color_temp']) / max(artwork_stats['color_temp'], target_stats['color_temp'])
    
    # Overall difference score (0-1 range)
    difference_score = (brightness_diff + saturation_diff + color_temp_diff) / 3
    
    # Calculate dynamic adjustment ranges based on difference
    # Higher difference = wider adjustment range allowed
    base_range = config.get('base_adjustment_range', 0.2)
    max_range = config.get('max_adjustment_range', 0.5)
    
    adjustment_range = base_range + (difference_score * (max_range - base_range))
    
    # Calculate dynamic min/max constraints
    min_adjustment = max(0.6, 1.0 - adjustment_range)  # Never go below 0.6
    max_adjustment = min(1.4, 1.0 + adjustment_range)  # Never go above 1.4
    
    # Determine color dominance
    r_mean, g_mean, b_mean = artwork_stats['r_mean'], artwork_stats['g_mean'], artwork_stats['b_mean']
    max_channel = max(r_mean, g_mean, b_mean)
    
    # Calculate purity/dominance of each channel (0-1 range)
    # Higher value means the channel is more dominant
    r_dominance = r_mean / max_channel if max_channel > 0 else 0
    g_dominance = g_mean / max_channel if max_channel > 0 else 0
    b_dominance = b_mean / max_channel if max_channel > 0 else 0
    
    # Calculate overall color purity/dominance score
    # This will be high when one channel significantly outweighs others
    color_purity = max(
        abs(r_dominance - g_dominance),
        abs(r_dominance - b_dominance),
        abs(g_dominance - b_dominance)
    )
    
    # Determine which channel is dominant
    dominant_channel = 'r' if r_mean == max_channel else ('g' if g_mean == max_channel else 'b')
    
    # Threshold for considering an image "color-dominant"
    is_color_dominant = color_purity > 0.3 and max_channel > 100
    is_moderate_dominant = color_purity > 0.15 and max_channel > 100
    
    # Special check for red content that might not trigger full dominance
    has_significant_red = r_mean > 70 and r_mean > g_mean * 1.3 and r_mean > b_mean * 1.3
    
    if is_color_dominant:
        logger.info(f"Detected color-dominant artwork (channel: {dominant_channel}, purity: {color_purity:.2f})")
        # Allow wider adjustment range for dominant colors to preserve their vibrancy
        max_adjustment = min(1.5, max_adjustment * 1.1)
    elif is_moderate_dominant and has_significant_red:
        logger.info(f"Detected moderate red dominance (purity: {color_purity:.2f})")
    
    # Calculate brightness adjustment - use weighted approach
    brightness_factor = 0.8 + 0.2 * (target_stats['brightness'] / artwork_stats['brightness'])
    
    # For color-dominant images, preserve brightness differently based on dominant channel
    if is_color_dominant:
        if dominant_channel == 'r':
            # Red-dominant: preserve more brightness
            brightness_factor = min(brightness_factor * 1.05, 1.0)
        elif dominant_channel == 'b':
            # Blue-dominant: more conservative brightness preservation
            brightness_factor = min(brightness_factor * 1.05, 1.0)
        else:  # green or other
            brightness_factor = min(brightness_factor * 1.05, 1.0)
    
    # Special handling for red text on white background in multiply mode
    if config.get('blend_mode') == 'multiply' and has_significant_red and not is_color_dominant:
        # Preserve more brightness for red text
        brightness_factor = min(max(brightness_factor, 0.94), 1.0)
    
    brightness_factor = max(min_adjustment, min(max_adjustment, brightness_factor))
    
    # Calculate saturation adjustment with more conservative approach
    if artwork_stats['saturation'] > 0.05:  # Avoid division by very small numbers
        # Calculate raw saturation factor
        raw_saturation_factor = target_stats['saturation'] / artwork_stats['saturation']
        
        # For multiply blend mode, approach saturation differently
        if config.get('blend_mode') == 'multiply':
            if raw_saturation_factor > 1.0:
                # Never increase saturation for multiply blend
                saturation_factor = 1.0
            else:
                # For decreases, use a gentler approach based on how vibrant the color is
                # More vibrant colors (higher saturation) get less reduction
                reduction_strength = max(0.6, min(0.9, 1.0 - artwork_stats['saturation']))
                
                # For color-dominant images, be even more conservative with saturation reduction
                if is_color_dominant:
                    if dominant_channel == 'r':
                        reduction_strength *= 0.7  # Very conservative for red
                    elif dominant_channel == 'b':
                        reduction_strength *= 0.85  # Less conservative for blue
                    else:
                        reduction_strength *= 0.8  # Moderate for green
                # Special case for red text/elements on white 
                elif has_significant_red:
                    reduction_strength *= 0.75  # Conservative for significant red elements
                    
                saturation_factor = 1.0 - ((1.0 - raw_saturation_factor) * reduction_strength)
        else:
            # For other blend modes, use the existing dampening logic
            if raw_saturation_factor > 1.0:
                saturation_factor = 1.0 + (raw_saturation_factor - 1.0) * 0.7
            else:
                saturation_factor = raw_saturation_factor
    else:
        saturation_factor = 1.0  # No adjustment for nearly monochrome images
    
    # For color-dominant images, adjust saturation differently based on dominant channel
    if is_color_dominant:
        if dominant_channel == 'r':
            saturation_factor = max(saturation_factor, 0.85)  # Lower minimum
            saturation_factor *= 1.05  # Reduced boost
        elif dominant_channel == 'b':
            saturation_factor = max(saturation_factor, 0.85)
            saturation_factor *= 1.02  # Very subtle boost for blue
        else:  # green
            saturation_factor = max(saturation_factor, 0.87)
            saturation_factor *= 1.05  # Moderate boost for green
    # Special handling for red text/elements on white background
    elif has_significant_red and config.get('blend_mode') == 'multiply':
        saturation_factor = max(saturation_factor, 0.88)
        saturation_factor *= 1.02  # Very subtle boost
    
    saturation_factor = max(min_adjustment, min(max_adjustment, saturation_factor))
    
    # Calculate color temperature adjustment
    temp_factor = target_stats['color_temp'] / (artwork_stats['color_temp'] if artwork_stats['color_temp'] > 0.1 else 0.1)
    
    # For multiply blend mode, handle temperature more subtly
    if config.get('blend_mode') == 'multiply':
        # Apply a very slight warming to all multiply blends
        # This helps with the white areas needing a bit more warmth
        temp_bias = 1.04  # Warmer bias (> 1.0 = warmer) - increased from 1.03 for better white areas
        
        # For blue-dominant images, use a stronger warming bias
        if is_color_dominant and dominant_channel == 'b':
            temp_bias = 1.05  # Stronger warming for blue
            
        temp_factor = temp_factor * 0.7 + temp_bias * 0.3
    
    # Calculate RGB adjustments to shift color temperature
    r_shift = 1.0
    g_shift = 1.0
    b_shift = 1.0
    
    if temp_factor > 1.05:  # Target is warmer
        # Use a gentler approach for warming
        warming_amount = (temp_factor - 1.0) * 0.7
        r_shift = 1.0 + warming_amount * 0.2
        g_shift = 1.0 + warming_amount * 0.1
        b_shift = max(min_adjustment, 1.0 - warming_amount * 0.2)
    elif temp_factor < 0.95:  # Target is cooler
        # Use a gentler approach for cooling
        cooling_amount = (1.0 - temp_factor) * 0.7
        r_shift = max(min_adjustment, 1.0 - cooling_amount * 0.2)
        b_shift = 1.0 + cooling_amount * 0.2
    
    # For color-dominant images, preserve the dominant channel
    if is_color_dominant:
        if dominant_channel == 'r':
            r_shift = max(r_shift, 1.0)  # Never reduce dominant red
            if r_shift == 1.0:
                r_shift = 1.05  # Slightly boost red
        elif dominant_channel == 'g':
            g_shift = max(g_shift, 1.0)  # Never reduce dominant green
            if g_shift == 1.0:
                g_shift = 1.05  # Slightly boost green
        elif dominant_channel == 'b':
            b_shift = max(b_shift, 1.0)  # Never reduce dominant blue
            if b_shift == 1.0:
                b_shift = 1.05  # Slightly boost blue
    
    # Special handling for red text on white in multiply mode
    if has_significant_red and config.get('blend_mode') == 'multiply' and not is_color_dominant:
        r_shift = max(r_shift, 1.01)  # Slight boost to red
    
    # Calculate tint adjustment based on dominant colors
    r_tint = target_stats['r_mean'] / 128
    g_tint = target_stats['g_mean'] / 128
    b_tint = target_stats['b_mean'] / 128
    
    # Normalize the tint values to preserve overall brightness
    tint_avg = (r_tint + g_tint + b_tint) / 3
    if tint_avg > 0:
        r_tint = r_tint / tint_avg
        g_tint = g_tint / tint_avg
        b_tint = b_tint / tint_avg
    
    # Calculate how different the artwork's color is from the background
    color_difference = (
        abs(artwork_stats['r_mean'] - target_stats['r_mean']) +
        abs(artwork_stats['g_mean'] - target_stats['g_mean']) +
        abs(artwork_stats['b_mean'] - target_stats['b_mean'])
    ) / (3 * 255)  # Normalize to 0-1 range
    
    # Use higher threshold for multiply blend mode
    if config.get('blend_mode') == 'multiply':
        tint_threshold = 0.4  # Even higher threshold for multiply blend
    else:
        tint_threshold = 0.25  # Standard high threshold
    
    max_tint = 0.1  # Very subtle maximum tint
    
    if color_difference < tint_threshold:
        tint_weight = 0  # No tint if the colors are already somewhat close
    else:
        # Use squared scaling for very subtle graduation
        tint_scale = ((color_difference - tint_threshold) / (1 - tint_threshold)) ** 2
        tint_weight = max_tint * tint_scale
    
    # For multiply blend mode, further reduce tint effect
    if config.get('blend_mode') == 'multiply':
        tint_weight *= 0.5  # Half the tint effect for multiply blend
    
    # For very bright areas, further reduce tinting
    if artwork_stats['brightness'] > 180:
        tint_weight *= 0.5  # Halve the tint effect for bright areas
    
    # For color-dominant images, reduce tint weight to preserve original color
    if is_color_dominant:
        # Reduce tint more for red than blue (red needs more preservation)
        if dominant_channel == 'r':
            tint_weight *= 0.6
        elif dominant_channel == 'b':
            tint_weight *= 0.7
        else:
            tint_weight *= 0.65
    # Special case for red text
    elif has_significant_red:
        tint_weight *= 0.7
        
    # Combine temperature shift with tint
    r_adjust = (r_shift * (1 - tint_weight)) + (r_tint * tint_weight)
    g_adjust = (g_shift * (1 - tint_weight)) + (g_tint * tint_weight)
    b_adjust = (b_shift * (1 - tint_weight)) + (b_tint * tint_weight)
    
    # Additional boost for dominant channel in color-dominant images
    if is_color_dominant:
        if dominant_channel == 'r':
            r_adjust *= 1.05  # Reduced boost (was 1.1)
            g_adjust *= 0.98  # Slightly reduce green to enhance red
            b_adjust *= 0.98  # Slightly reduce blue to enhance red
        elif dominant_channel == 'g':
            g_adjust *= 1.08  # Boost green to preserve vibrancy
            r_adjust *= 0.98  # Slightly reduce red to enhance green
            b_adjust *= 0.98  # Slightly reduce blue to enhance green
        elif dominant_channel == 'b':
            b_adjust *= 1.06  # More modest boost for blue (compared to red)
            r_adjust *= 0.98  # Slightly reduce red to enhance blue
            g_adjust *= 0.98  # Slightly reduce green to enhance blue
    # Special handling for red text on white background
    elif has_significant_red and config.get('blend_mode') == 'multiply':
        r_adjust *= 1.03  # Subtle boost for red channel
    
    # For multiply blend mode, enforce a warmer white balance - REGARDLESS of color dominance
    if config.get('blend_mode') == 'multiply':
        # Create a consistent warm adjustment (more red, slightly more green, less blue)
        warm_r = 1.06  # Boost red (increased from 1.05 for better white warmth) 
        warm_g = 1.03  # Slightly boost green (increased from 1.02)
        warm_b = 0.94  # Reduce blue (reduced from 0.95 for better white warmth)
        
        # For blue-dominant images, apply even stronger warming to counteract the coolness
        if is_color_dominant and dominant_channel == 'b':
            warm_r = 1.07  # Stronger red boost for blue images
            warm_g = 1.03  # Stronger green boost for blue images
            warm_b = 0.93  # Stronger blue reduction for blue images
        
        # Mix these warm adjustments with the calculated RGB values
        # Use a stronger influence for very bright areas
        if artwork_stats['brightness'] > 220:  # Very bright whites
            r_adjust = r_adjust * 0.3 + warm_r * 0.7  # Stronger warming for very bright whites
            g_adjust = g_adjust * 0.3 + warm_g * 0.7
            b_adjust = b_adjust * 0.3 + warm_b * 0.7
        elif artwork_stats['brightness'] > 200:
            r_adjust = r_adjust * 0.4 + warm_r * 0.6
            g_adjust = g_adjust * 0.4 + warm_g * 0.6
            b_adjust = b_adjust * 0.4 + warm_b * 0.6
        elif artwork_stats['brightness'] > 150:
            r_adjust = r_adjust * 0.6 + warm_r * 0.4
            g_adjust = g_adjust * 0.6 + warm_g * 0.4
            b_adjust = b_adjust * 0.6 + warm_b * 0.4
        else:
            # For darker areas, use a lighter touch with warming
            r_adjust = r_adjust * 0.8 + warm_r * 0.2
            g_adjust = g_adjust * 0.8 + warm_g * 0.2
            b_adjust = b_adjust * 0.8 + warm_b * 0.2
    
    # Improved channel balancing for multiply blend mode
    if config.get('blend_mode') == 'multiply':
        # For multiply blend, ensure all channels are at least 0.95 to prevent color casts
        r_adjust = max(0.95, r_adjust)
        g_adjust = max(0.95, g_adjust)
        b_adjust = max(0.95, b_adjust)
        
        # If any channel is significantly lower, bring them closer together
        min_adjust = min(r_adjust, g_adjust, b_adjust)
        max_adjust = max(r_adjust, g_adjust, b_adjust)
        
        if max_adjust - min_adjust > 0.05:
            # Calculate a balanced adjustment that preserves relative relationships
            # but reduces the extreme differences
            avg_adjust = (r_adjust + g_adjust + b_adjust) / 3
            
            # Move each adjustment closer to the average
            # But use different balancing for different colors
            if is_color_dominant:
                if dominant_channel == 'r':
                    balance_factor = 0.5  # Less aggressive balancing for red
                elif dominant_channel == 'b':
                    balance_factor = 0.7  # More aggressive balancing for blue
                else:
                    balance_factor = 0.6  # Standard balancing for other colors
            else:
                # For red text on white, use less balancing to preserve red
                if has_significant_red:
                    balance_factor = 0.5  # Less balancing for red text
                else:
                    balance_factor = 0.6  # Standard balancing for non-dominant colors
                
            r_adjust = r_adjust * (1 - balance_factor) + avg_adjust * balance_factor
            g_adjust = g_adjust * (1 - balance_factor) + avg_adjust * balance_factor
            b_adjust = b_adjust * (1 - balance_factor) + avg_adjust * balance_factor
    else:
        # For non-multiply blends, use the existing balancing code
        avg_adjust = (r_adjust + g_adjust + b_adjust) / 3
        if abs(r_adjust - avg_adjust) > 0.1 or abs(g_adjust - avg_adjust) > 0.1 or abs(b_adjust - avg_adjust) > 0.1:
            # Move each adjustment closer to the average
            # But use different balancing for different colors
            if is_color_dominant:
                if dominant_channel == 'r':
                    balance_factor = 0.2  # Less aggressive balancing for red
                elif dominant_channel == 'b':
                    balance_factor = 0.3  # Standard balancing for blue
                else:
                    balance_factor = 0.25  # Moderate balancing for other colors
            else:
                # For red text on white, use less balancing to preserve red
                if has_significant_red:
                    balance_factor = 0.2  # Less balancing for red text
                else:
                    balance_factor = 0.3  # Standard balancing for non-dominant colors
                
            r_adjust = r_adjust * (1 - balance_factor) + avg_adjust * balance_factor
            g_adjust = g_adjust * (1 - balance_factor) + avg_adjust * balance_factor
            b_adjust = b_adjust * (1 - balance_factor) + avg_adjust * balance_factor
    
    # Clamp the channel adjustments
    r_adjust = max(min_adjustment, min(max_adjustment, r_adjust))
    g_adjust = max(min_adjustment, min(max_adjustment, g_adjust))
    b_adjust = max(min_adjustment, min(max_adjustment, b_adjust))
    
    # Increase contrast slightly if adding to a high-contrast background
    contrast_factor = 1.0
    if target_stats['contrast'] > 100:  # High contrast background
        contrast_factor = 1.1  # Subtle increase
    
    # For color-dominant images, adjust contrast differently based on dominant channel
    if is_color_dominant:
        if dominant_channel == 'r':
            contrast_factor *= 1.03  # Additional contrast to make reds pop
        elif dominant_channel == 'b':
            contrast_factor *= 1.01  # Very subtle contrast for blues
        else:
            contrast_factor *= 1.02  # Moderate contrast for other colors
    
    # For monochrome artwork being placed on colorful backgrounds
    # Add a slight tint matching the background
    color_factor = 1.0
    if artwork_stats['is_monochrome'] and not target_stats['is_monochrome']:
        color_factor = 1.1  # Slightly increase colorfulness (reduced from 1.2)
    
    # If artwork is much more colorful than background, tone it down a bit more
    if artwork_stats['colorfulness'] > target_stats['colorfulness'] * 2:
        # But less aggressively for color-dominant images
        if is_color_dominant:
            if dominant_channel == 'r':
                saturation_factor *= 0.92  # Less reduction for reds
            elif dominant_channel == 'b':
                saturation_factor *= 0.85  # Standard reduction for blues
            else:
                saturation_factor *= 0.88  # Moderate reduction for other colors
        else:
            saturation_factor *= 0.85  # Standard saturation reduction
    
    # Calculate overall vibrancy adjustment (combination of saturation and contrast)
    vibrancy_factor = (saturation_factor + contrast_factor) / 2
    
    # For color-dominant images, adjust vibrancy based on dominant channel
    if is_color_dominant:
        if dominant_channel == 'r':
            vibrancy_factor *= 1.02  # Reduced vibrancy boost for reds
        elif dominant_channel == 'b':
            vibrancy_factor *= 1.02  # Subtle vibrancy boost for blues
        else:
            vibrancy_factor *= 1.03  # Balanced vibrancy for other colors
    
    logger.info(f"Calculated adjustments: brightness={brightness_factor:.2f}, "
                f"saturation={saturation_factor:.2f}, contrast={contrast_factor:.2f}, "
                f"R={r_adjust:.2f}, G={g_adjust:.2f}, B={b_adjust:.2f}, "
                f"dominant={dominant_channel if is_color_dominant else 'none'}")
    
    return {
        'brightness': brightness_factor,
        'saturation': saturation_factor,
        'contrast': contrast_factor,
        'color': color_factor,
        'r_adjust': r_adjust,
        'g_adjust': g_adjust,
        'b_adjust': b_adjust,
        'vibrancy': vibrancy_factor,
        'is_color_dominant': is_color_dominant,  # Include this for reference
        'dominant_channel': dominant_channel,    # Include this for reference
        'has_significant_red': has_significant_red,  # Include this for reference
        'applied_constraints': {
            'min_adjustment': min_adjustment,
            'max_adjustment': max_adjustment,
            'difference_score': difference_score
        }
    }
    
def apply_adaptive_adjustments(artwork, adjustments):
    """
    Apply calculated color adjustments to the artwork.
    
    Args:
        artwork: PIL Image of the artwork
        adjustments: Dictionary of adjustment factors
    
    Returns:
        PIL Image with adjustments applied
    """
    try:
        # Convert to proper color space for adjustments
        if artwork.mode != 'RGBA':
            artwork = artwork.convert('RGBA')
        
        # Split into bands
        r, g, b, a = artwork.split()
        
        # Create RGB image for adjustments
        rgb_img = Image.merge('RGB', (r, g, b))
        
        # Apply brightness adjustment
        enhancer = ImageEnhance.Brightness(rgb_img)
        adjusted_rgb = enhancer.enhance(adjustments['brightness'])
        
        # Apply contrast adjustment
        enhancer = ImageEnhance.Contrast(adjusted_rgb)
        adjusted_rgb = enhancer.enhance(adjustments['contrast'])
        
        # Apply saturation adjustment
        enhancer = ImageEnhance.Color(adjusted_rgb)
        adjusted_rgb = enhancer.enhance(adjustments['saturation'])
        
        # Apply channel-specific adjustments for color temperature and tint
        r, g, b = adjusted_rgb.split()
        
        # Adjust individual channels if needed
        if abs(adjustments['r_adjust'] - 1.0) > 0.01:
            r_enhancer = ImageEnhance.Brightness(r)
            r = r_enhancer.enhance(adjustments['r_adjust'])
            
        if abs(adjustments['g_adjust'] - 1.0) > 0.01:
            g_enhancer = ImageEnhance.Brightness(g)
            g = g_enhancer.enhance(adjustments['g_adjust'])
            
        if abs(adjustments['b_adjust'] - 1.0) > 0.01:
            b_enhancer = ImageEnhance.Brightness(b)
            b = b_enhancer.enhance(adjustments['b_adjust'])
        
        # Merge channels back together
        adjusted_rgb = Image.merge('RGB', (r, g, b))
        
        # Recombine with alpha
        result = Image.new('RGBA', artwork.size)
        result.paste(adjusted_rgb, mask=a)
        
        return result
    except Exception as e:
        logger.error(f"Error applying adaptive adjustments: {str(e)}")
        return artwork  # Return original artwork if adjustment fails

# Main function to use in the warp_artwork pipeline
def adaptive_color_match(artwork_image, background_image, transform_points, blend_mode='normal', opacity=1.0):
    """
    Adapt artwork colors to match the background where it will be placed.
    Memory-optimized version.
    """
    try:
        # Extract the target region from background
        target_region = extract_surrounding_region(background_image, transform_points)
        
        # Analyze color statistics
        artwork_stats = analyze_image_colors(artwork_image)
        target_stats = analyze_image_colors(target_region)
        
        # Free memory for target_region as we no longer need it
        del target_region
        gc.collect()
        
        # Log color analysis results
        logger.info(f"Artwork color stats: brightness={artwork_stats['brightness']:.1f}, "
                    f"saturation={artwork_stats['saturation']:.2f}, "
                    f"color_temp={artwork_stats['color_temp']:.2f}")
        
        logger.info(f"Target region stats: brightness={target_stats['brightness']:.1f}, "
                    f"saturation={target_stats['saturation']:.2f}, "
                    f"color_temp={target_stats['color_temp']:.2f}")
        
        # Setup config for blend mode
        config = {'blend_mode': blend_mode} if blend_mode else {}
        
        # Calculate adjustments based on the analysis
        adjustments = calculate_adjustments(artwork_stats, target_stats, config)
        
        # Apply adjustments to the artwork
        adjusted_artwork = apply_adaptive_adjustments(artwork_image, adjustments)
        
        # Free original artwork memory
        del artwork_image
        gc.collect()
        
        # If blend_mode is multiply, also apply a specialized multiply effect
        if blend_mode.lower() == 'multiply':
            logger.info(f"Applying additional multiply blend effect with opacity {opacity}")
            
            # Memory optimization: Process in chunks for large images
            # Check image size and use chunking if necessary
            width, height = adjusted_artwork.size
            total_pixels = width * height
            
            # If image is large, use chunked processing
            if total_pixels > 3000000:  # Threshold for large images (3 megapixels)
                logger.info(f"Large image detected ({width}x{height}), using chunked processing")
                return apply_multiply_effect_chunked(adjusted_artwork, opacity, adjustments)
            else:
                # For smaller images, use the regular approach
                return apply_multiply_effect(adjusted_artwork, opacity, adjustments)
        
        return adjusted_artwork
        
    except Exception as e:
        logger.error(f"Error in adaptive color matching: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fall back to original artwork
        logger.info("Falling back to original artwork without color adjustments")
        return artwork_image

def apply_multiply_effect(artwork, opacity, adjustments):
    """Apply multiply blend effect to an image."""
    try:
        # Convert to numpy array for pixel manipulation
        artwork_array = np.array(artwork, dtype=np.float32)
        
        # Separate RGB and alpha channels
        artwork_rgb = artwork_array[:, :, :3]
        artwork_alpha = artwork_array[:, :, 3]
        
        # Check if it's color-dominant
        is_color_dominant = adjustments.get('is_color_dominant', False)
        dominant_channel = adjustments.get('dominant_channel', None)
        has_significant_red = adjustments.get('has_significant_red', False)
        
        # Dynamic darkening factor based on color dominance
        if is_color_dominant:
            if dominant_channel == 'r':
                # Use slightly more darkening for red-dominant images
                base_darkening = 0.78  # Was 0.8 (higher = less darkening)
            elif dominant_channel == 'b':
                # Use standard darkening for blue-dominant (blues handle darkening better)
                base_darkening = 0.74
            else:
                # Use moderate darkening for other colors
                base_darkening = 0.75
        elif has_significant_red:
            # For red text on white
            base_darkening = 0.72  # Slightly stronger darkening for red text
        else:
            # Standard approach for non-dominant colors
            base_darkening = 0.73
        
        logger.info(f"Using base darkening factor {base_darkening} for multiply blend")
        
        # Create pixel-wise darkening factors based on dominant channel
        max_values = np.max(artwork_rgb, axis=2, keepdims=True)
        channel_dominance = artwork_rgb / (max_values + 0.001)  # How close each channel is to the max (0-1)
        
        # Different boost amount for different color channels
        r_boost = 0.17  # Stronger boost for red channel
        g_boost = 0.12  # Moderate boost for green channel
        b_boost = 0.10  # Lower boost for blue channel
        
        # For images with red text or elements on white
        if has_significant_red and not is_color_dominant:
            r_boost = 0.19  # Even stronger red preservation
        
        # Apply channel-specific boosts to create darkening factors
        darkening_factors = np.zeros_like(artwork_rgb)
        darkening_factors[:,:,0] = base_darkening + (r_boost * channel_dominance[:,:,0])  # Red channel
        darkening_factors[:,:,1] = base_darkening + (g_boost * channel_dominance[:,:,1])  # Green channel
        darkening_factors[:,:,2] = base_darkening + (b_boost * channel_dominance[:,:,2])  # Blue channel
        
        # Apply darkening with channel-specific factors
        darkened_rgb = artwork_rgb * darkening_factors
        
        # Apply opacity to the darkening effect
        if opacity < 1.0:
            darkened_rgb = darkened_rgb * opacity + artwork_rgb * (1.0 - opacity)
        
        # Free memory for intermediate arrays
        del artwork_rgb, max_values, channel_dominance, darkening_factors
        gc.collect()
        
        # Create final array with darkened RGB and original alpha
        final_array = np.zeros_like(artwork_array)
        final_array[:, :, :3] = darkened_rgb
        final_array[:, :, 3] = artwork_alpha  # Keep original alpha
        
        # Free memory for more intermediate arrays
        del darkened_rgb, artwork_alpha
        gc.collect()
        
        # Clip values and convert back to uint8
        final_array = np.clip(final_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(final_array)
    
    except Exception as e:
        logger.error(f"Error applying multiply effect: {str(e)}")
        # Return original as fallback
        return artwork

def apply_multiply_effect_chunked(artwork, opacity, adjustments):
    """Apply multiply blend effect to an image using chunked processing to save memory."""
    try:
        # Get image dimensions
        width, height = artwork.size
        
        # Define chunk size (rows)
        chunk_size = 500  # Process 500 rows at a time
        
        # Create a new empty image to hold the result
        result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Check if it's color-dominant
        is_color_dominant = adjustments.get('is_color_dominant', False)
        dominant_channel = adjustments.get('dominant_channel', None)
        has_significant_red = adjustments.get('has_significant_red', False)
        
        # Determine base darkening factor (same logic as before)
        if is_color_dominant:
            if dominant_channel == 'r':
                base_darkening = 0.77
            elif dominant_channel == 'b':
                base_darkening = 0.7
            else:
                base_darkening = 0.75
        elif has_significant_red:
            base_darkening = 0.72
        else:
            base_darkening = 0.7
        
        # Different boost amount for different color channels
        r_boost = 0.17
        g_boost = 0.12
        b_boost = 0.10
        
        # For images with red text or elements on white
        if has_significant_red and not is_color_dominant:
            r_boost = 0.19
            
        logger.info(f"Using chunked processing with chunk size {chunk_size} rows")
        
        # Process in chunks
        for y_start in range(0, height, chunk_size):
            # Calculate end of current chunk
            y_end = min(y_start + chunk_size, height)
            
            # Crop current chunk
            chunk = artwork.crop((0, y_start, width, y_end))
            
            # Convert to numpy array
            chunk_array = np.array(chunk, dtype=np.float32)
            
            # Process chunk using same logic as before
            chunk_rgb = chunk_array[:, :, :3]
            chunk_alpha = chunk_array[:, :, 3]
            
            # Create pixel-wise darkening factors
            max_values = np.max(chunk_rgb, axis=2, keepdims=True)
            channel_dominance = chunk_rgb / (max_values + 0.001)
            
            # Apply channel-specific boosts
            darkening_factors = np.zeros_like(chunk_rgb)
            darkening_factors[:,:,0] = base_darkening + (r_boost * channel_dominance[:,:,0])
            darkening_factors[:,:,1] = base_darkening + (g_boost * channel_dominance[:,:,1])
            darkening_factors[:,:,2] = base_darkening + (b_boost * channel_dominance[:,:,2])
            
            # Apply darkening
            darkened_rgb = chunk_rgb * darkening_factors
            
            # Apply opacity
            if opacity < 1.0:
                darkened_rgb = darkened_rgb * opacity + chunk_rgb * (1.0 - opacity)
            
            # Create final array for this chunk
            final_chunk_array = np.zeros_like(chunk_array)
            final_chunk_array[:, :, :3] = darkened_rgb
            final_chunk_array[:, :, 3] = chunk_alpha
            
            # Clip values and convert back to uint8
            final_chunk_array = np.clip(final_chunk_array, 0, 255).astype(np.uint8)
            
            # Convert back to PIL image
            processed_chunk = Image.fromarray(final_chunk_array)
            
            # Paste this chunk into the result image
            result.paste(processed_chunk, (0, y_start))
            
            # Clean up memory for this chunk
            del chunk, chunk_array, chunk_rgb, chunk_alpha, max_values
            del channel_dominance, darkening_factors, darkened_rgb, final_chunk_array, processed_chunk
            gc.collect()
            
            # Log progress for large images
            logger.info(f"Processed chunk {y_start}-{y_end} of {height} rows ({int((y_end/height)*100)}% complete)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in chunked multiply effect: {str(e)}")
        # Return original as fallback
        return artwork

def warp_artwork(artwork_data, transform_points, output_size, template_name, 
                 smart_object_name, aspect_ratio, record_id, background_data=None, 
                 blend_mode='normal', opacity=255):
    """Warp artwork to fit transform points and apply adaptive color matching - memory optimized"""
    try:
        # Load the artwork
        artwork = Image.open(artwork_data).convert("RGBA")
        logger.info(f"Artwork dimensions: {artwork.size}")
        
        # Load background if provided, for color matching
        background = None
        if background_data:
            try:
                background = Image.open(background_data).convert("RGBA")
                logger.info(f"Background dimensions: {background.size}")
            except Exception as bg_error:
                logger.error(f"Error loading background for color matching: {str(bg_error)}")
                
        # Calculate dimensions from transform points
        # Ensure transform_points are properly formatted
        if isinstance(transform_points, list) and transform_points and isinstance(transform_points[0], list):
            transform_points = [(point[0], point[1]) for point in transform_points]
            
        x_coords = [p[0] for p in transform_points]
        y_coords = [p[1] for p in transform_points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        target_width = int(max_x - min_x)
        target_height = int(max_y - min_y)
        
        logger.info(f"Target dimensions: {target_width}x{target_height}")
        
        # Resize artwork to fit the dimensions while preserving aspect ratio
        artwork_resized = ImageOps.contain(artwork, (target_width, target_height))
        logger.info(f"Resized artwork to: {artwork_resized.size}")
        
        # Free original artwork memory
        del artwork
        gc.collect()
        
        # Apply adaptive color matching if we have a background
        if background:
            # Convert opacity from 0-255 to 0-1 for adaptive_color_match
            opacity_normalized = opacity / 255.0
            
            logger.info(f"Applying adaptive color matching with blend mode: {blend_mode}, opacity: {opacity_normalized}")
            
            # Apply the color matching
            artwork_processed = adaptive_color_match(
                artwork_resized, 
                background, 
                transform_points,
                blend_mode,
                opacity_normalized
            )
            
            # Free memory for the resized artwork as we now have the processed version
            del artwork_resized
            gc.collect()
            
            logger.info("Adaptive color matching applied")
        else:
            # Without background, just use the resized artwork
            artwork_processed = artwork_resized
            
            # If multiply blend mode, apply simple darkening
            if blend_mode.lower() == 'multiply':
                logger.info(f"No background available, applying simple multiply darkening with opacity {opacity/255.0}")
                dummy_background = Image.new('RGBA', artwork_processed.size, (128, 128, 128, 255))
                artwork_processed = adaptive_color_match(
                    artwork_processed,
                    dummy_background,
                    [(0, 0), (artwork_processed.width, 0), (artwork_processed.width, artwork_processed.height), (0, artwork_processed.height)],
                    'multiply',
                    opacity/255.0
                )
                # Free memory for the dummy background
                del dummy_background
                gc.collect()
        
        # Create source points from the artwork dimensions
        source_points = [
            (0, 0),                                      # Top-left
            (artwork_processed.width, 0),                  # Top-right
            (artwork_processed.width, artwork_processed.height),  # Bottom-right
            (0, artwork_processed.height)                  # Bottom-left
        ]
        
        # Apply perspective transform for precise placement
        logger.info("Applying perspective transform...")
        warped_artwork = apply_perspective_transform(
            artwork_processed,
            source_points,
            transform_points,
            output_size
        )
        
        # Free memory for the processed artwork as we now have the warped version
        del artwork_processed
        if background:
            del background
        gc.collect()
        
        # Save to BytesIO object and optimize memory
        output_buffer = BytesIO()
        warped_artwork.save(output_buffer, format='PNG', optimize=True)
        output_buffer.seek(0)
        
        # Free memory for the warped artwork
        del warped_artwork
        gc.collect()
        
        # Generate S3 key for the warped artwork
        sanitized_template_name = template_name.replace(' ', '_').lower()
        sanitized_aspect_ratio = aspect_ratio.replace(':', '_')
        output_key = f"warped-artworks/{sanitized_template_name}_{smart_object_name}_{sanitized_aspect_ratio}_{record_id}.png"
        
        # Upload to S3
        s3_url = s3_handler.upload_file_obj(
            output_buffer,
            output_key,
            content_type='image/png'
        )
        
        return s3_url
    except Exception as e:
        logger.error(f"Error warping artwork: {str(e)}")
        logger.error(traceback.format_exc())
        return None
        
def process_mockup(template_name, backgrounds, coordinates, blend_modes, artworks, record_id):
    """Process mockup using template backgrounds and coordinates with adaptive color matching"""
    results = {
        'warped_artworks': {},
        'mockup_urls': {},
        'aspect_ratios_processed': []
    }
    
    # Filter artworks dictionary to only include non-empty values
    valid_artworks = {k: v for k, v in artworks.items() if v}
    logger.info(f"Valid artworks: {list(valid_artworks.keys())}")
    
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
                # Extract blend modes from the same JSON if available
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
            
            # Download background image once for all artworks in this aspect ratio
            background_data = s3_handler.download_file(background_url)
            
            # Load background and check size
            bg_image = Image.open(background_data)
            width, height = bg_image.width, bg_image.height
            
            # Check if background is very large and needs downscaling for processing
            # We'll still use the original background URL for Renderform
            total_pixels = width * height
            MAX_PROCESSING_PIXELS = 9000000  # ~9 megapixels is a good balance
            
            # If background is too large, scale it down for processing
            if total_pixels > MAX_PROCESSING_PIXELS:
                scale_factor = (MAX_PROCESSING_PIXELS / total_pixels) ** 0.5
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                logger.info(f"Background is very large ({width}x{height}), downscaling to {new_width}x{new_height} for processing")
                
                # Scale down the background for memory efficiency
                bg_image = bg_image.resize((new_width, new_height), Image.LANCZOS)
                
                # Also scale down the transform points
                scaled_coord_data = {}
                for obj_name, points in coord_data.items():
                    scaled_points = []
                    for point in points:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            scaled_points.append((point[0] * scale_factor, point[1] * scale_factor))
                        else:
                            # If point format is unexpected, skip this object
                            logger.warning(f"Invalid point format in {obj_name}, skipping: {point}")
                            scaled_points = []
                            break
                    
                    if scaled_points:
                        scaled_coord_data[obj_name] = scaled_points
                
                # Use scaled coordinates
                original_coord_data = coord_data
                coord_data = scaled_coord_data
                
                # Reset the background data for further processing
                bg_buffer = BytesIO()
                bg_image.save(bg_buffer, format='PNG')
                bg_buffer.seek(0)
                background_data = bg_buffer
            
            output_size = (bg_image.width, bg_image.height)
            
            # Rewind background data for future use
            background_data.seek(0)
            
            # Process each artwork
            warped_artworks = {}
            for smart_obj_name, artwork_url in valid_artworks.items():
                if smart_obj_name not in coord_data:
                    logger.warning(f"No coordinates for {smart_obj_name} in {aspect_ratio}, skipping")
                    continue
                
                # Get transform points from coordinates
                transform_points = coord_data[smart_obj_name]
                
                # Get blend info for this smart object
                blend_info = aspect_blend_modes.get(smart_obj_name, {})
                blend_mode = blend_info.get('blend_mode', 'normal')
                opacity = blend_info.get('opacity', 255)
                
                logger.info(f"Processing {smart_obj_name} with blend mode: {blend_mode}, opacity: {opacity}")
                
                # Download artwork
                artwork_data = s3_handler.download_file(artwork_url)
                
                # Make a fresh copy of background data for each artwork
                background_copy = BytesIO(background_data.getvalue())
                
                try:
                    # Process artwork with blend mode and color matching
                    warped_url = warp_artwork(
                        artwork_data,
                        transform_points,
                        output_size,
                        template_name,
                        smart_obj_name,
                        aspect_ratio,
                        record_id,
                        background_copy,  # Pass background for color matching
                        blend_mode,
                        opacity
                    )
                    
                    if warped_url:
                        warped_artworks[smart_obj_name] = warped_url
                        logger.info(f"Successfully processed {smart_obj_name}: {warped_url}")
                except Exception as e:
                    logger.error(f"Error processing {smart_obj_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Clean up
                del artwork_data, background_copy
                gc.collect()
            
            # Clean up background resources
            del background_data, bg_image
            gc.collect()
            
            # Create final mockup using Renderform
            if warped_artworks:
                try:
                    mockup_result = renderform_handler.create_mockup(
                        aspect_ratio,
                        background_url,
                        warped_artworks
                    )
                    if mockup_result and 'href' in mockup_result:
                        results['mockup_urls'][aspect_ratio] = mockup_result['href']
                        results['aspect_ratios_processed'].append(aspect_ratio)
                        logger.info(f"Created final mockup for {aspect_ratio}: {mockup_result['href']}")
                except Exception as e:
                    logger.error(f"Error creating final mockup for {aspect_ratio}: {str(e)}")
            
            # Store warped artwork URLs
            if aspect_ratio not in results['warped_artworks']:
                results['warped_artworks'][aspect_ratio] = {}
            results['warped_artworks'][aspect_ratio].update(warped_artworks)
            
            logger.info(f"Successfully processed mockup for {aspect_ratio}")
        except Exception as e:
            logger.error(f"Error processing mockup for aspect ratio {aspect_ratio}: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Format response for Airtable
    airtable_updates = {}
    
    # Add warped artwork URLs to update fields
    for aspect_ratio, artworks_dict in results['warped_artworks'].items():
        for print_name, url in artworks_dict.items():
            field_name = f"{print_name} ({aspect_ratio})"
            airtable_updates[field_name] = url
    
    # Add mockup URLs to update fields
    for aspect_ratio, url in results['mockup_urls'].items():
        airtable_updates[f"Ad Mockup ({aspect_ratio})"] = url
    
    # Update status
    if results['aspect_ratios_processed']:
        process_status = f"Success: Processed {len(results['aspect_ratios_processed'])} mockups"
    else:
        process_status = "Failed: No mockups could be processed"
    
    airtable_updates['Upload Status'] = process_status
    
    return {
        'airtable_updates': airtable_updates,
        'detailed_results': results
    }
