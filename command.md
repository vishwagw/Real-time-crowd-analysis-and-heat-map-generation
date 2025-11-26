I'll create a comprehensive crowd analysis AI application for restaurants using computer vision in Python. This will include person detection, counting, heatmap generation, and occupancy analytics.I've created a comprehensive crowd analysis AI application for restaurants with these key features:

## Core Features:

1. **Person Detection**
   - YOLO v3 for accurate detection (with HOG fallback)
   - Real-time tracking and counting
   - Bounding boxes and center point tracking

2. **Analytics Dashboard**
   - Live people count
   - Occupancy level (Low/Moderate/High)
   - Zone-based counting (entrance, dining, exit)
   - Historical trend graph
   - Real-time heatmap showing busy areas

3. **Zone Management**
   - Customizable zones for different restaurant areas
   - Per-zone people counting
   - Visual zone overlays

4. **Data Export**
   - Save analytics snapshots
   - JSON format for easy integration

## Setup Requirements:

```bash
pip install opencv-python numpy
```

**For YOLO (recommended):**
Download these files and place in the same directory:
- yolov3.weights (from YOLO official site)
- yolov3.cfg
- coco.names

**Or use the built-in HOG detector** (no downloads needed, but less accurate)

## Usage:

```python
# For webcam
analyzer = RestaurantCrowdAnalyzer(video_source=0, max_capacity=50)

# For video file
analyzer = RestaurantCrowdAnalyzer(video_source='restaurant.mp4', max_capacity=50)

analyzer.run()
```

**Controls:**
- Press 'q' to quit
- Press 's' to save analytics

Would you like me to add specific features like customer flow analysis, wait time estimation, or integration with a database for long-term tracking?

# Using YOLOV8

I'll update the application to use YOLOv8, which is more modern and efficient than v3!Perfect! I've updated the application to use **YOLOv8** with enhanced features:

## Key Improvements with YOLOv8:

### **Better Performance**
- Faster inference speed
- More accurate person detection
- Lower computational requirements
- Built-in class filtering

### **New Features Added**

1. **Object Tracking**
   - Assigns unique IDs to each person
   - Tracks movement across frames
   - Monitors entry/exit flow

2. **Entry/Exit Counting**
   - Automatic detection of people entering/exiting
   - Net flow calculation
   - Entry and exit line visualization

3. **Enhanced Heatmap**
   - Smooth Gaussian blur
   - Better visual representation
   - Real-time density mapping

4. **Comprehensive Analytics**
   - Peak count tracking
   - Average count over time
   - Zone-based distribution
   - Occupancy percentage bar
   - Trend graph

5. **Interactive Controls**
   - `q` - Quit
   - `s` - Save analytics
   - `r` - Reset counters
   - `h` - Toggle heatmap

## Installation:

```bash
pip install ultralytics opencv-python numpy
```

## Usage:

```python
# Basic usage
analyzer = RestaurantCrowdAnalyzer(
    video_source=0,           # Webcam
    max_capacity=50,
    model_path='yolov8n.pt'   # Nano model (fastest)
)

# For better accuracy, use larger models:
# 'yolov8s.pt' - Small
# 'yolov8m.pt' - Medium  
# 'yolov8l.pt' - Large
# 'yolov8x.pt' - Extra Large

analyzer.run(show_heatmap=True, save_video=False)
```

## Model Selection:
- **yolov8n.pt** - Fastest, good for real-time
- **yolov8s.pt** - Balanced
- **yolov8m.pt** - More accurate, slightly slower

The model will auto-download on first run! Would you like me to add features like customer dwell time analysis, table occupancy tracking, or integration with a database?