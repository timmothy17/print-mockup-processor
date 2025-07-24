import os
import logging
import traceback
from flask import Flask, request, jsonify
import json
import gc  # For garbage collection
from processor import process_mockup
from airtable_handler import AirtableHandler
from config import SECRET_KEY, DEBUG, PORT
import os
import logging
import traceback
from flask import Flask, request, jsonify
import json
import gc  # For garbage collection
from processor import process_mockup, ArtworkProcessor, PhotoshopColorAdjustments
from airtable_handler import AirtableHandler
from config import SECRET_KEY, DEBUG, PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Initialize handlers
try:
    airtable_handler = AirtableHandler()
    logger.info("AirtableHandler initialized successfully")
except Exception as e:
    logger.error(f"Error initializing AirtableHandler: {str(e)}")
    logger.error(traceback.format_exc())
    # Define a fallback handler in case of error
    class FallbackAirtableHandler:
        def __init__(self):
            logger.warning("Using fallback AirtableHandler implementation")
            
        def update_record(self, record_id, fields):
            logger.warning(f"Fallback AirtableHandler: Called update_record for {record_id}")
            return {"status": "fallback_implementation", "fields": fields}
    
    # Initialize with fallback
    airtable_handler = FallbackAirtableHandler()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Force garbage collection before reporting health
    gc.collect()
    
    return jsonify({
        'status': 'healthy',
        'message': 'Mockup processor service is running (v2.0 - Photoshop Blend Modes)',
        'memory_usage_mb': get_memory_usage_mb(),
        'features': {
            'photoshop_blend_modes': True,
            'vectorized_processing': True,
            'color_adjustments': True,
            'tiled_processing': True
        }
    })

@app.route('/process-template', methods=['POST'])
def process_template():
    """Process PSD template and artworks - LEGACY ENDPOINT"""
    logger.warning("Legacy process-template endpoint called - redirecting to /process-mockup")
    return process_mockup_handler()

@app.route('/process-mockup', methods=['POST'])
def process_mockup_handler():
    """Process mockup with Photoshop blend modes and color adjustments"""
    try:
        # Force garbage collection before processing
        gc.collect()
        
        # Get request data
        data = request.json
        logger.info(f"Received request for Photoshop blend mode processing")
        
        # Extract required fields
        template_name = data.get('template_name', 'unnamed_template')
        backgrounds = data.get('backgrounds', {})
        coordinates = data.get('coordinates', {})
        blend_modes = data.get('blend_modes', {})
        artworks = data.get('artworks', {})
        record_id = data.get('record_id')
        
        # Extract optional parameters
        use_local_mockup = data.get('use_local_mockup', True)
        intelligent_scaling = data.get('intelligent_scaling', True)
        
        # NEW: Extract white backings and manual adjustments
        white_backings = data.get('white_backings', {})
        manual_adjustments_json = data.get('manual_adjustments_json')
        
        # Log processing details
        logger.info(f"Processing mockup for {template_name} (Record ID: {record_id})")
        logger.info(f"Using local mockup generation: {use_local_mockup}")
        logger.info(f"Using intelligent scaling: {intelligent_scaling}")
        logger.info(f"White backings provided: {len(white_backings)} aspect ratios")
        
        # Log manual adjustments status
        if manual_adjustments_json and manual_adjustments_json.strip():
            logger.info("Manual color adjustments provided - will use user-defined values")
        else:
            logger.info("No manual adjustments - will generate default adjustment JSON")
        
        # Validate request
        if not record_id:
            return jsonify({'error': 'No record_id provided'}), 400
        
        if not backgrounds:
            return jsonify({'error': 'No backgrounds provided'}), 400
            
        if not coordinates:
            return jsonify({'error': 'No coordinates provided'}), 400
        
        # Check for white backings (recommended for best results)
        if not white_backings:
            logger.warning("No white backings provided - results may not be optimal")
            
        # Process mockup using the updated function with Photoshop blend modes
        start_time = get_current_time_ms()
        results = process_mockup(
            template_name, 
            backgrounds, 
            coordinates, 
            blend_modes, 
            artworks, 
            record_id,
            use_local_mockup=use_local_mockup,
            intelligent_scaling=intelligent_scaling,
            manual_adjustments_json=manual_adjustments_json,
            white_backings=white_backings
        )
        processing_time = (get_current_time_ms() - start_time) / 1000.0  # Convert to seconds
        
        # Update Airtable
        airtable_error = None
        try:
            airtable_handler.update_record(record_id, results['airtable_updates'])
            logger.info(f"Updated Airtable record {record_id}")
            logger.info(results['airtable_updates'])
        except Exception as e:
            logger.error(f"Error updating Airtable: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Convert the exception to a string safely
            airtable_error = f"{type(e).__name__}: {str(e)}"
            
            # We'll still return success if processing succeeded but Airtable update failed
            results['airtable_update_error'] = airtable_error
        
        # Add processing time to the response
        results['processing_time_seconds'] = processing_time
        
        # Force garbage collection
        gc.collect()
        
        # Construct the response
        response_data = {
            'success': True,
            'message': "Mockup processing complete with Photoshop blend modes",
            'processing_time_seconds': processing_time,
            'results': results,
            'version': '2.0',
            'features_used': {
                'photoshop_multiply': True,
                'color_adjustments': manual_adjustments_json is not None,
                'white_backing': len(white_backings) > 0,
                'vectorized_processing': True
            }
        }
        
        # If there was an Airtable error, include it in the response
        if airtable_error:
            response_data['airtable_update_error'] = airtable_error
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Ensure the error message is a string
        error_message = f"{type(e).__name__}: {str(e)}"
        
        return jsonify({
            'success': False,
            'error': error_message,
            'message': "Failed to process mockup with Photoshop blend modes",
            'version': '2.0'
        }), 500

@app.route('/mockup-info', methods=['GET'])
def mockup_info():
    """Get information about the mockup processor"""
    return jsonify({
        'version': '2.0.0',
        'name': 'Photoshop Blend Mode Processor',
        'features': {
            'photoshop_multiply_blend': True,
            'photoshop_luminosity_blend': True,
            'vectorized_processing': True,
            'tiled_memory_management': True,
            'color_adjustments': {
                'scale': '-100 to +100 (Photoshop style)',
                'adjustments': [
                    'brightness', 'contrast', 'saturation', 'vibrance',
                    'highlights', 'shadows', 'whites', 'blacks',
                    'exposure', 'gamma', 'hue_shift', 'warmth', 
                    'tint', 'clarity', 'structure'
                ]
            },
            'blend_modes': ['multiply', 'luminosity', 'normal'],
            'intelligent_scaling': True,
            'white_backing_support': True
        },
        'memory_usage_mb': get_memory_usage_mb()
    })

@app.route('/generate-default-adjustments', methods=['POST'])
def generate_default_adjustments():
    """Generate default color adjustment JSON for given artwork names"""
    try:
        data = request.json
        smart_object_names = data.get('smart_object_names', [])
        
        if not smart_object_names:
            return jsonify({'error': 'No smart_object_names provided'}), 400
        
        color_adjustments = PhotoshopColorAdjustments()
        adjustments_json = color_adjustments.create_adjustments_json(smart_object_names)
        
        return jsonify({
            'success': True,
            'adjustments_json': adjustments_json,
            'smart_objects_count': len(smart_object_names)
        })
        
    except Exception as e:
        logger.error(f"Error generating default adjustments: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/test-blend-modes', methods=['POST'])
def test_blend_modes():
    """Test endpoint for blend mode functionality"""
    try:
        from processor import test_blend_modes, test_photoshop_adjustments
        
        # Run tests
        blend_test_result = test_blend_modes()
        adjustment_test_result = test_photoshop_adjustments()
        
        return jsonify({
            'success': True,
            'blend_modes_test': blend_test_result,
            'adjustments_test': adjustment_test_result is not None,
            'memory_usage_mb': get_memory_usage_mb()
        })
        
    except Exception as e:
        logger.error(f"Error in blend mode tests: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug-file', methods=['POST'])
def debug_file():
    """Debug endpoint to check file processing"""
    try:
        # Import these here to avoid startup failures if not available
        from PIL import Image
        from psd_tools import PSDImage
        
        # Get the file from the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Try to open the file with Pillow
        try:
            image = Image.open(file)
            pillow_info = {
                'format': image.format,
                'size': image.size,
                'mode': image.mode
            }
        except Exception as e:
            pillow_info = {'error': str(e)}
            
        # Try to open with psd-tools
        file.seek(0)
        try:
            psd = PSDImage.open(file)
            psd_info = {
                'width': psd.width,
                'height': psd.height,
                'layer_count': len(list(psd.descendants()))
            }
            
            # Get layer info
            layers = []
            for layer in psd.descendants():
                if hasattr(layer, 'name'):
                    layer_info = {
                        'name': layer.name,
                        'kind': layer.kind if hasattr(layer, 'kind') else 'unknown',
                        'visible': layer.visible if hasattr(layer, 'visible') else None,
                        'is_group': layer.is_group() if hasattr(layer, 'is_group') else False
                    }
                    layers.append(layer_info)
            
            psd_info['layers'] = layers[:20]  # Limit to first 20 layers
            
        except Exception as e:
            psd_info = {'error': str(e)}
            
        return jsonify({
            'filename': file.filename,
            'pillow_info': pillow_info,
            'psd_info': psd_info,
            'processor_version': '2.0'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

def get_current_time_ms():
    """Get current time in milliseconds"""
    import time
    return time.time() * 1000

if __name__ == '__main__':
    # Log startup information
    logger.info(f"Starting Ad Mockup Processor v2.0 (Photoshop Blend Modes)")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Port: {PORT}")
    logger.info(f"Initial memory usage: {get_memory_usage_mb():.2f}MB")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=DEBUG
    )
