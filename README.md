# Ad Mockup Generator

A sophisticated automated system for generating product mockups with intelligent color matching, Photoshop-style blend modes, and perspective transformation. Built for high-volume production workflows with Airtable integration.

## 🎯 What It Does

This system takes product artwork and automatically places it onto mockup templates (like t-shirts, posters, etc.) with realistic perspective warping and intelligent color blending. It's designed for print-on-demand and marketing teams who need to generate hundreds of mockups quickly.

**Input:** Product artwork + PSD mockup template
**Output:** Photorealistic product mockups in multiple aspect ratios

## ✨ Key Features

### 🎨 Advanced Color Processing
- **Photoshop-Style Blend Modes**: Multiply and luminosity blending with vectorized processing
- **Adaptive Color Adjustments**: 15 different adjustment types (brightness, contrast, saturation, warmth, etc.)
- **Color Analysis Pipeline**: Automatic analysis of artwork colors for optimal blending
- **Smart Opacity Control**: Per-layer opacity with alpha compositing

### 🔄 Intelligent Image Processing
- **Perspective Transformation**: Warps artwork to match mockup perspective using OpenCV
- **Smart Object Coordination**: Extracts placement coordinates from PSD files
- **Multi-Aspect Ratio Support**: Generates 1:1, 4:5, and 9:16 versions simultaneously
- **Memory-Optimized Processing**: Handles large images (6000px+) with tiled processing

### 🏗️ Production-Ready Architecture
- **Scalable Processing**: Chunked processing for memory efficiency
- **S3 Integration**: Automatic file storage and URL generation
- **Airtable Automation**: Complete workflow integration
- **Error Recovery**: Comprehensive error handling and fallbacks

## 🔬 Technical Evolution

### Version 1.0 - Basic Color Matching
The initial version used simple color analysis to determine blend parameters:
- Basic RGB analysis of artwork and background regions
- Simple brightness/contrast adjustments
- Manual blend mode selection

### Version 1.5 - Intelligent Color Analysis
Enhanced the system with sophisticated color analysis:
- **Surrounding Region Analysis**: Extracted colors from areas around the mockup placement
- **Color Temperature Detection**: Automatic warm/cool color analysis
- **Adaptive Blend Strength**: Dynamic opacity based on color contrast
- **Monochrome Detection**: Special handling for black/white artwork

### Version 2.0 - Photoshop Integration (Current)
Complete rewrite with professional-grade image processing:
- **Vectorized Blend Modes**: NumPy-based multiply and luminosity blending
- **Photoshop-Style Adjustments**: 15 adjustment types with -100 to +100 scales
- **White Backing System**: Professional multiply blend workflow
- **Tiled Processing**: Memory-efficient processing of large images
- **Global Settings**: Configurable luminosity blend strength

## 🏛️ Architecture
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Airtable      │───▶│   Flask API      │───▶│   S3 Storage    │
│   Automation    │    │   (app.py)       │    │   (Final Files) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
│
▼
┌──────────────────────┐
│   ArtworkProcessor   │
│   (processor.py)     │
└──────────────────────┘
│
┌───────────────────┼───────────────────┐
▼                   ▼                   ▼
┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  Geometric   │  │  TiledBlend     │  │  PhotoshopColor  │
│  Processor   │  │  Processor      │  │  Adjustments     │
└──────────────┘  └─────────────────┘  └──────────────────┘

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- AWS S3 bucket for file storage
- Airtable account (optional)

### Installation

1. **Clone the repository**
```
git clone https://github.com/yourusername/ad-mockup-generator.git
cd ad-mockup-generator
```

2. **Install dependencies**
```
pip install -r requirements.txt
```

3.Set up environment variables
```
cp .env.example .env
# Edit .env with your actual credentials
```

4. Run the application
```
python app.py
```

The API will be available at http://localhost:5000

📡 API Reference
Core Endpoints
- POST /process-mockup
  - Generates mockups with advanced color processing.
  - Request Body:
    json{
      "template_name": "T-Shirt Mockup",
      "backgrounds": {
        "1:1": "https://s3.amazonaws.com/bucket/background-1x1.png",
        "4:5": "https://s3.amazonaws.com/bucket/background-4x5.png"
      },
      "coordinates": {
        "1:1": "{\"coordinates\": {\"Print1\": [[100,100], [200,100], [200,200], [100,200]]}, \"blend_modes\": {\"Print1\": {\"blend_mode\": \"multiply\", \"opacity\": 255}}}"
      },
      "artworks": {
        "Print1": "https://s3.amazonaws.com/bucket/artwork.png"
      },
      "white_backings": {
        "1:1": "https://s3.amazonaws.com/bucket/white-backing-1x1.png"
      },
      "manual_adjustments_json": "{\"Print1\": {\"brightness\": 10, \"contrast\": 5}}",
      "record_id": "rec123456",
      "use_local_mockup": true,
      "intelligent_scaling": true
    }
    Response:
    json{
      "success": true,
      "message": "Mockup processing complete with Photoshop blend modes",
      "processing_time_seconds": 45.2,
      "results": {
        "mockup_urls": {
          "1:1": "https://s3.amazonaws.com/bucket/final-mockup.jpg"
        },
        "warped_artworks": {
          "1:1": {
            "Print1": "https://s3.amazonaws.com/bucket/warped-artwork.png"
          }
        }
      }
    }
- GET /health
  - System health check with memory usage.

- GET /mockup-info
  - System capabilities and version information.

- POST /generate-default-adjustments
  - Generates default color adjustment JSON for artwork layers.


## 🎨 Color Adjustment System
**The system uses Photoshop-style adjustments with a -100 to +100 scale:**

- Basic Adjustments:
  - Brightness: Overall brightness (-100=black, +100=white)
  - Contrast: Overall contrast (-100=flat, +100=maximum)
  - Exposure: Exposure compensation

- Color Adjustments:
  -  Saturation: Color intensity (-100=grayscale, +100=hyper-saturated)
  -  Vibrance: Smart saturation (protects skin tones)
  -  Hue Shift: Hue rotation (-100 to +100)
  -  Warmth: Color temperature (-100=cooler/blue, +100=warmer/orange)
  -  Tint: Green-magenta shift

- Tone Adjustments:
  - Highlights: Bright area adjustment
  - Shadows: Dark area adjustment
  - Whites: White point adjustment
  - Blacks: Black point adjustment

- Enhancement:
  - Clarity: Midtone contrast
  - Structure: Fine detail enhancement

- Example Adjustment JSON
  - json{
    "_global_settings": {
      "luminosity_blend_strength": 20
    },
    "Print1": {
      "brightness": 10,
      "contrast": 5,
      "saturation": -10,
      "warmth": 15,
      "shadows": 20
    }
  }

### 🔧 Configuration
#### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `S3_BUCKET_NAME` | AWS S3 bucket for file storage | `my-mockup-bucket` |
| `AWS_REGION` | AWS region | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | AWS access key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `secret...` |
| `AIRTABLE_API_KEY` | Airtable API key | `key...` |
| `AIRTABLE_BASE_ID` | Airtable base ID | `app...` |
| `AIRTABLE_TABLE_NAME` | Airtable table name | `Ad Mockups` |
| `RENDERFORM_API_KEY` | Renderform API key (optional) | `rf_...` |
| `RENDERFORM_TEMPLATE_ID` | Default Renderform template ID | `tpl_...` |
| `RENDERFORM_TEMPLATE_ID_1_1` | 1:1 aspect ratio template ID | `tpl_...` |
| `RENDERFORM_TEMPLATE_ID_4_5` | 4:5 aspect ratio template ID | `tpl_...` |
| `RENDERFORM_TEMPLATE_ID_9_16` | 9:16 aspect ratio template ID | `tpl_...` |
| `SECRET_KEY` | Flask secret key | `your-secret-key-here` |
| `DEBUG` | Flask debug mode | `False` |
| `PORT` | Application port | `5000` |


🎯 Use Cases
Print-on-Demand
  - Automatically generate t-shirt, poster, and merchandise mockups.
  - Handle high-volume artwork processing.
  - Multiple aspect ratios for different platforms

Marketing Teams
  - Create product visualizations for campaigns.
  - A/B test different color treatments.
  - Batch process artwork collections

E-commerce
  - Generate product images for online stores.
  - Consistent mockup styling across catalogs.
  - Automated workflow integration

📊 Performance
  - Memory Optimization: Tiled processing handles 6000px+ images
  - Processing Speed: ~45 seconds for full multi-aspect ratio generation
  - File Sizes: Intelligent scaling produces 8-10MB final files
  - Quality: 98% JPEG quality with 300 DPI output

🤝 Contributing
  - Fork the repository
  - Create a feature branch (git checkout -b feature/amazing-feature)
  - Commit your changes (git commit -m 'Add amazing feature')
  - Push to the branch (git push origin feature/amazing-feature)
  - Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Built with Python, PIL, OpenCV, and NumPy
Inspired by professional print production workflows
Thanks to the open-source image processing community


For questions or support, please open an issue on GitHub.

## 2. .env.example

```bash
# AWS S3 Configuration
S3_BUCKET_NAME=your-s3-bucket-name
AWS_REGION=your-aws-region
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key

# Airtable Configuration
AIRTABLE_API_KEY=your-airtable-api-key
AIRTABLE_BASE_ID=your-airtable-base-id
AIRTABLE_TABLE_NAME=Your Table Name

# Renderform Configuration (Optional)
RENDERFORM_API_KEY=your-renderform-api-key
RENDERFORM_TEMPLATE_ID=your-default-template-id
RENDERFORM_TEMPLATE_ID_1_1=your-1-1-template-id
RENDERFORM_TEMPLATE_ID_4_5=your-4-5-template-id
RENDERFORM_TEMPLATE_ID_9_16=your-9-16-template-id

# Flask Configuration
SECRET_KEY=your-secret-key-here-make-it-long-and-random
DEBUG=False
PORT=5000
```
