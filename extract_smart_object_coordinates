"""
PSD Smart Object Coordinate Extraction

This module extracts smart object coordinates and blend mode information 
from Adobe Photoshop PSD files for use with the mockup engine.
"""

import logging
import json
from psd_tools import PSDImage
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PSDJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle PSD-specific objects"""
    def default(self, obj):
        if hasattr(obj, 'name'):  # Handle BlendMode enums
            return obj.name.lower()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def find_smart_objects(psd, prefix="Print"):
    """Find all smart objects with a specific prefix"""
    smart_objects = {}
    for layer in psd.descendants():
        if (hasattr(layer, 'name') and 
            layer.name.startswith(prefix) and 
            hasattr(layer, 'kind') and 
            layer.kind == 'smartobject'):
            smart_objects[layer.name] = layer
            logger.info(f"Found smart object: {layer.name}")
    return smart_objects


def get_transform_points(smart_object):
    """Extract transform points from smart object if available"""
    if (hasattr(smart_object, 'smart_object') and 
        hasattr(smart_object.smart_object, 'transform_box')):
        # Transform box contains 4 points (x,y pairs) defining the corners
        transform_box = smart_object.smart_object.transform_box
        
        # Convert to a list of 4 (x,y) tuples
        if len(transform_box) == 8:  # Should be 8 coordinates (4 points)
            return [
                (transform_box[0], transform_box[1]),  # Top-left
                (transform_box[2], transform_box[3]),  # Top-right
                (transform_box[4], transform_box[5]),  # Bottom-right
                (transform_box[6], transform_box[7])   # Bottom-left
            ]
    
    # Fallback: Use the layer's bounding box
    return [
        (smart_object.left, smart_object.top),  # Top-left
        (smart_object.right, smart_object.top),  # Top-right
        (smart_object.right, smart_object.bottom),  # Bottom-right
        (smart_object.left, smart_object.bottom)  # Bottom-left
    ]


def extract_smart_object_blend_data(psd):
    """Extract blend mode information for all smart objects"""
    blend_data = {}
    
    for layer in psd.descendants():
        if (hasattr(layer, 'name') and 
            layer.name.startswith('Print') and 
            hasattr(layer, 'kind') and 
            layer.kind == 'smartobject'):
            
            # Convert BlendMode enum to string
            blend_mode = 'normal'
            if hasattr(layer, 'blend_mode'):
                # Extract the enum name (e.g., 'MULTIPLY' from BlendMode.MULTIPLY)
                if hasattr(layer.blend_mode, 'name'):
                    blend_mode = layer.blend_mode.name.lower()
                else:
                    blend_mode = str(layer.blend_mode).split('.')[-1].lower()
            
            blend_info = {
                'blend_mode': blend_mode,
                'opacity': getattr(layer, 'opacity', 255),
                'fill_opacity': getattr(layer, 'fill_opacity', 255),
                'visible': getattr(layer, 'visible', True)
            }
            
            # Check if the layer has any special blending flags
            if hasattr(layer, 'flags'):
                blend_info['flags'] = {
                    'transparency_protected': getattr(layer.flags, 'transparency_protected', False),
                    'clipping': getattr(layer.flags, 'clipping', False)
                }
            
            blend_data[layer.name] = blend_info
            logger.info(f"{layer.name} blend mode: {blend_info['blend_mode']}, opacity: {blend_info['opacity']}")
    
    return blend_data


def extract_smart_object_coordinates(psd_path):
    """
    Extract coordinates and blend mode information for all Print smart objects
    
    Args:
        psd_path: Path to the PSD file
        
    Returns:
        Dictionary containing dimensions, coordinates, and blend modes
    """
    logger.info(f"Opening PSD file: {psd_path}")
    
    try:
        # Load the PSD file
        psd = PSDImage.open(psd_path)
        logger.info(f"PSD dimensions: {psd.width}x{psd.height}")
        
        # Find all Print smart objects
        smart_objects = find_smart_objects(psd, "Print")
        
        if not smart_objects:
            logger.warning("No smart objects with prefix 'Print' found")
            # Try to find any smart objects as a fallback
            all_smart_objects = find_smart_objects(psd, "")
            if all_smart_objects:
                logger.info(f"Found {len(all_smart_objects)} smart objects without 'Print' prefix")
                available_layers = [layer.name for layer in all_smart_objects.values()]
                logger.info(f"Available smart objects: {', '.join(available_layers)}")
            else:
                logger.warning("No smart objects found at all")
                available_layers = [layer.name for layer in psd.descendants() if hasattr(layer, 'name')]
                logger.info(f"Available layers: {', '.join(available_layers[:20])}...")
            return {"error": "No 'Print' smart objects found"}
        
        # Extract coordinates for each smart object
        coordinates = {}
        dimensions = {"width": psd.width, "height": psd.height}
        
        for name, smart_obj in smart_objects.items():
            transform_points = get_transform_points(smart_obj)
            logger.info(f"{name} transform points: {transform_points}")
            coordinates[name] = transform_points
        
        # Extract blend mode data
        blend_modes = extract_smart_object_blend_data(psd)
        
        # Format the result
        result = {
            "dimensions": dimensions,
            "coordinates": coordinates,
            "blend_modes": blend_modes
        }
        
        logger.info(f"Successfully extracted coordinates and blend modes for {len(coordinates)} smart objects")
        return result
        
    except Exception as e:
        logger.error(f"Error processing PSD file: {e}")
        logger.exception(e)
        return {"error": str(e)}


def process_psd_file(psd_path, output_path=None, print_json=True):
    """
    Process a PSD file and extract smart object coordinates and blend modes
    
    Args:
        psd_path: Path to the PSD file
        output_path: Optional path to save JSON output (default: same name as PSD with .json extension)
        print_json: Whether to print the JSON output to console
        
    Returns:
        Dictionary with extraction results
    """
    result = extract_smart_object_coordinates(psd_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return result
    
    # Convert to JSON string
    json_output = json.dumps(result, indent=2, cls=PSDJSONEncoder)
    
    # Save to file if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(json_output)
        logger.info(f"Results saved to {output_path}")
    
    # Print JSON output for copying
    if print_json:
        print("\n" + "="*50)
        print("JSON OUTPUT (Copy this for use with mockup engine):")
        print("="*50)
        print(json_output)
        print("="*50 + "\n")
    
    return result


def extract_coordinates_only(psd_path):
    """
    Simple function to extract just the coordinates JSON string
    
    Args:
        psd_path: Path to the PSD file
        
    Returns:
        JSON string ready for use with mockup engine
    """
    result = extract_smart_object_coordinates(psd_path)
    
    if "error" in result:
        return None
    
    return json.dumps(result, indent=2, cls=PSDJSONEncoder)


# Example usage
if __name__ == "__main__":
    # Example: Extract coordinates from a PSD file
    # Replace with your actual PSD file path
    psd_file = "path/to/your/template.psd"
    
    # This will print the JSON output you need
    process_psd_file(psd_file)
    
    # Or get just the JSON string
    # json_string = extract_coordinates_only(psd_file)
