# Contributing to Ad Mockup Generator

Thank you for considering contributing to this project! This document provides guidelines and information for contributors.

## 🎯 Project Vision

This project aims to provide professional-grade automated mockup generation with:
- Production-ready performance and reliability
- Advanced image processing capabilities
- Easy integration with existing workflows
- Clear, maintainable code

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
```
git clone https://github.com/yourusername/ad-mockup-generator.git
cd ad-mockup-generator
```

2. **Set up development environment**
```
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment**
```
bashcp .env.example .env
# Edit .env with your development credentials
```

4. **Run tests (when available)**
```
bashpython -m pytest tests/
```


### 📝 Code Style
Python Guidelines

Follow PEP 8 style guidelines
Use type hints where appropriate
Comprehensive docstrings for classes and functions
Meaningful variable and function names

Example Function Documentation
```
pythondef process_single_artwork(self, artwork_data, transform_points, output_size, 
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
```

## 🏗️ Architecture
### Core Components
- processor.py
  - ArtworkProcessor: Main processing pipeline
  - GeometricProcessor: Perspective transformations
  - TiledBlendProcessor: Memory-efficient blending
  - PhotoshopColorAdjustments: Color adjustment system

### Key Design Principles
- Memory Efficiency: Use tiled processing for large images
- Error Recovery: Comprehensive exception handling
- Modularity: Clear separation of concerns
- Performance: Vectorized operations where possible

### 🎨 Contributing to Image Processing
- Color Adjustment System
- The color adjustment system uses Photoshop-style parameters (-100 to +100):

```
def convert_scale_to_factor(value, adjustment_type):
    """Convert Photoshop-style -100 to +100 scale to multiplication factors."""
    if adjustment_type in ['brightness', 'exposure']:
        return 1.0 + (value / 100.0)
    elif adjustment_type in ['contrast', 'clarity']:
        if value > 0:
            return 1.0 + (value / 100.0) * 2.0
        else:
            return 1.0 + (value / 100.0) * 0.9
    # ... more adjustment types
```

### Blend Mode Implementation
Blend modes use vectorized NumPy operations:
```
pythondef multiply_blend(top_rgba, bottom_rgba):
    """Apply Photoshop multiply blend mode using vectorized operations."""
    
    top_norm = top_rgba.astype(np.float32) / 255.0
    bottom_norm = bottom_rgba.astype(np.float32) / 255.0
    
    multiply_rgb = top_norm[:, :, :3] * bottom_norm[:, :, :3]
    # ... alpha compositing logic
    
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)
```

🐛 Bug Reports
When reporting bugs, please include:
- Environment information:
- Python version, Operating system, Package versions
- Steps to reproduce: Input data (if possible), Configuration used, Expected vs actual behavior
- Logs and error messages
- Full stack traces
- Relevant log output

✨ Feature Requests
For new features, please:
- Check existing issues to avoid duplicates
- Describe the use case and problem being solved
- Propose a solution if you have ideas
- Consider backwards compatibility

🔧 Areas for Contribution
- High Priority
- Performance optimization for large image processing
- Additional blend modes (overlay, soft light, etc.)
- Unit tests for core processing functions
- Documentation improvements

Medium Priority
- Docker containerization
- CLI interface for standalone use
- Batch processing improvements
- Error recovery enhancements

Advanced
- GPU acceleration with CUDA/OpenCL
- Machine learning color matching
- Advanced color space support (CMYK, LAB)

📋 Pull Request Process
- Create a feature branch
```
bashgit checkout -b feature/your-feature-name
```

- Make your changes
  Follow the code style guidelines. Add tests if applicable. Update documentation
  
- Test your changes
  Ensure existing functionality still works. Test with different image sizes and formats

- Commit with clear messages
```
bashgit commit -m "Add luminosity blend mode optimization
```

- Implement vectorized luminosity blending
- Reduce memory usage by 40%
- Add opacity parameter support"

Push and create PR
```
bashgit push origin feature/your-feature-name
```
PR Requirements
  - Clear description of changes
  - Link to related issues
  - Screenshots/examples if UI changes



# 🧪 Testing
- Manual Testing
```
bash# Test basic functionality
curl -X POST http://localhost:5000/health
```

- Test mockup generation (requires valid data)
```
curl -X POST http://localhost:5000/process-mockup \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

- Performance Testing
  - Test with large images (4000px+)
  - Monitor memory usage during processing
  - Verify output quality and file sizes

- 📚 Learning Resources
  - Image Processing
  - PIL/Pillow Documentation
  - OpenCV Python Tutorials
  - NumPy Documentation

- Photoshop Blend Modes
  - Adobe Blend Mode Reference
  - Blend Mode Mathematics

- 💬 Communication
  - GitHub Issues: Bug reports and feature requests
  - Discussions: General questions and ideas
  - Pull Requests: Code contributions and reviews

- 📄 License
By contributing to this project, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to make this project better! 🎉

## 4. LICENSE
MIT License
Copyright (c) 2025 [Your Name]
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
