# Print Mockup Processor
**Lightweight, RAM-scalable mockup processing engine for Photoshop-prepped assets.**


## 🎯 Motivation & Problem

### **The Photoshop Scripting Limitation**

While Photoshop scripting can automate mockup generation, it can become impractical for production use:

- **Local File Management**: Hundreds of artwork files need to be downloaded and managed locally
- **Database Disconnect**: No easy way to connect Photoshop scripts to modern databases and workflows
- **Resource Intensive**: Photoshop requires significant RAM and processing power for each operation and requires a device with Photoshop installed
- **Enterprise API Barrier**: Adobe's Creative SDK API requires enterprise licensing with a minimum spend of $150,000+ USD
- **No Cloud Integration**: Difficult to integrate with cloud storage and modern web workflows

### **The Real Challenge**

Traditional bulk mockup generation creates a painful workflow:

1. **Manual Photoshop Work**: Each variation requires individual attention
2. **File Management Nightmare**: Downloading, organizing, and processing hundreds of local files
3. **Database Integration Gap**: No straightforward way to connect Photoshop to Airtable, Google Sheets, or modern databases
4. **Resource Constraints**: Photoshop's memory requirements make bulk processing expensive
5. **No API Access**: Adobe's enterprise API is cost-prohibitive for most businesses

### **The Solution: Extract Photoshop's Power**

This engine solves the core problem by **extracting Photoshop's essential capabilities** into a lightweight, API-driven system:

#### **What Was Preserved from Photoshop:**
- ✅ **Exact multiply blend modes** - Same mathematical formulas as Photoshop
- ✅ **Perspective transformation** - Professional-quality warping and placement
- ✅ **Color adjustment system** - 15 Photoshop-style adjustments (-100 to +100 scale)
- ✅ **Smart object coordination** - Precise placement using extracted coordinates

#### **What Was Improved:**
- 🚀 **Direct database integration** - Works with any platform that can send API requests
- 🚀 **Cloud-native processing** - Direct image URLs, no local file management
- 🚀 **Memory efficient** - Runs on 512MB RAM or scales to high-performance servers
- 🚀 **True bulk processing** - Generate hundreds of mockups automatically
- 🚀 **Affordable API access** - No $150K enterprise licensing required

### **Key Innovation: The Three-Asset System**

The breakthrough is separating **creative work** (done once in Photoshop) from **production work** (automated at scale):

**Photoshop Prep** (Creative - Done Once):
- Design mockup template with transparent cutout
- Create color-corrected backing for realistic blending
- Extract smart object coordinates

**Automated Processing** (Production - Done Thousands of Times):
- Send artwork URL via API
- Get professional mockup back instantly
- Perfect color matching and perspective

### **Built for the Art Prints Industry**

This approach is **optimized for rectangular print placement** - perfect for:
- Art prints and posters
- T-shirt and apparel designs  
- Product mockups for e-commerce
- Marketing material variations
- Print-on-demand businesses

Instead of fighting Photoshop's limitations, we extracted its core strengths into a system designed for modern, database-driven workflows. You get Photoshop-quality results with the convenience of a simple API call.

## ✨ Key Features

### 🎨 Advanced Image Processing
- **Photoshop-Style Blend Modes**: Multiply and luminosity blending with vectorized NumPy processing
- **Adaptive Color Adjustments**: 15 different adjustment types (brightness, contrast, saturation, warmth, etc.)
- **Intelligent Color Analysis**: Automatic artwork color analysis for optimal blending
- **Smart Opacity Control**: Per-layer opacity with alpha compositing

### 🔄 Geometric Transformation
- **Perspective Transformation**: Warps artwork to match mockup perspective using OpenCV
- **Smart Object Coordination**: Extracts placement coordinates from PSD files
- **Multi-Aspect Ratio Support**: Generates 1:1, 4:5, and 9:16 versions simultaneously
- **Edge Smoothing**: Anti-aliasing for professional results

### 🏗️ Production-Ready Architecture
- **Scalable Processing**: Chunked processing for memory efficiency (512MB - 16GB+ RAM)
- **Vectorized Operations**: NumPy-based processing for maximum performance
- **Error Recovery**: Comprehensive error handling and fallbacks
- **Memory Optimization**: Aggressive cleanup and garbage collection

## 🔧 How It Works

### **The Three-Asset System**

The engine requires three pre-processed assets to generate mockups:

#### **1. Background Template**
- Your mockup background with a **transparent cutout** where artwork will be placed
- Can include overlays like reflections, shadows, or textures
- Overlay elements should have reduced opacity in the PNG for realism
- Example: T-shirt mockup with transparent area where design goes

#### **2. Color-Corrected Backing**
- A white backing that's been **color-corrected in Photoshop** to match the background
- This is the secret sauce - the backing captures the lighting and color environment
- The engine uses **multiply blend mode** on this backing to make artwork colors look realistic
- Essential for proper color integration with the background

#### **3. Original Artwork**
- Your product design or artwork to be placed
- Can be any resolution - the engine handles scaling and perspective transformation
- Works best with **direct image links** (AWS S3, Google Cloud, etc.)

### **Processing Pipeline**

1. **Perspective Transform**: Warps artwork to match mockup perspective using OpenCV
2. **Multiply Blend**: Applies Photoshop-style multiply blend with the color-corrected backing
3. **Luminosity Refinement**: Optional secondary blend for color enhancement (default 20% strength)
4. **Color Adjustments**: 15 Photoshop-style adjustments (brightness, contrast, etc.)
5. **Final Compositing**: Layers everything with the background template

### **Memory Efficiency**
- **Tiled Processing**: Large images processed in small chunks to minimize RAM usage
- **Garbage Collection**: Aggressive memory cleanup between operations
- **Intelligent Scaling**: Adapts processing based on available system resources
- **Scalable Performance**: From 512MB to 16GB+ RAM automatically

## 🎨 Photoshop Workflow Guide

### Setting Up Your Template

#### Create Background with Transparent Cutout
- Design your mockup background (t-shirt, poster frame, etc.)
- Use layer masks to create transparent area where artwork will be placed
- Add realistic overlays (reflections, shadows, fabric texture)
- Reduce overlay opacity (20-40%) for subtle realism

#### Color-Correct the Backing
- Create a white layer beneath your background
- Apply color correction to match the background's lighting environment
- Consider ambient light color, shadows, and surface properties
- This backing is crucial for realistic color blending

#### Extract Smart Object Coordinates
- Use the included PSD coordinate extractor (`utils/psd_extractor.py`)
- Identifies placement points for perspective transformation
- Exports JSON with exact coordinates and blend mode settings

### Why This Approach Works
- **Separates creative from production**: Design once, generate thousands
- **Maintains Photoshop quality**: Same blend modes and color science
- **Optimized for rectangular prints**: Perfect for art prints, posters, designs
- **Industry-proven**: Based on professional print production workflows

## 🎯 Color Adjustment System
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

## 🔗 Integration Examples

This engine works with any platform that can send API requests:

### Database + Automation Platforms
- **Airtable** (my production setup) - Visual interface with automation scripts
- **Monday.com** - Project management with custom workflows
- **Google Sheets** - Simple spreadsheet-driven automation
- **Notion** - Database with API integration
- **Zapier/Make** - Connect to hundreds of other tools

### E-commerce Platforms
- **Shopify** - Automated product mockup generation
- **WooCommerce** - WordPress integration via custom plugins
- **Etsy** - Bulk mockup creation for listings


🏗️ Architecture
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Database      │───▶│   MockupEngine   │───▶│   Final Mockup │
│   (Airtable,    │    │   (Core Pipeline)│    │   (S3/Local)    │
│   Sheets, etc.) │    └──────────────────┘    └─────────────────┘
└─────────────────┘             │
                                ▼
                    ┌──────────────────────┐
                    │   Processing Modules │
                    └──────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐
    │  Geometric   │  │  Photoshop      │  │  Memory-Efficient│
    │  Processor   │  │  Blend Modes    │  │  Compositor      │
    └──────────────┘  └─────────────────┘  └──────────────────┘

## 🧩 Core Components

- `core/mockup_engine.py` – Main processing pipeline with transforms and adjustments  
- `core/mockup_compositor.py` – Memory-efficient mockup composition & scaling  
- `core/layer_assembly.py` – Compositing and layer logic  
- `utils/psd_extractor.py` – PSD smart object coordinate extraction

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
  - Processing Speed: ~30 seconds per image generation (at 512 MB RAM and 0.5 CPU)
  - File Sizes: Intelligent scaling produces 8-10MB final files
  - Quality: 98% JPEG quality with 300 DPI output


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

Built with Python, PIL, OpenCV, and NumPy
Inspired by professional print production workflows
Thanks to the open-source image processing community


For questions or support, please open an issue on GitHub.

