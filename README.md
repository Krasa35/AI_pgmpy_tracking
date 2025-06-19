# Probabilistic Multi-Object Tracking with Bayesian Networks

A probabilistic object tracking system using Bayesian Networks to track pedestrians and cyclists across video frames.

## Overview

The system uses a Bayesian Network that considers four key features:
- **Position**: Spatial proximity between detections
- **Size**: Bounding box area similarity  
- **Color**: Histogram-based comparison using OpenCV
- **Velocity**: Movement patterns and direction consistency

**Note**: Bayesian temporal connections between frames are not fully implemented. Each frame is processed independently.

## Usage

```bash
python main.py <data_folder>
```

For debug mode with visual display, modify `main.py`:
```python
main(debug=True)
```

## Input Format

- `data_folder/bboxes.txt`: Detection file
- `data_folder/frames/`: Frame images directory

## Output

Space-separated IDs for each frame:
- `0, 1, 2, ...`: Continuing tracks (detection index)
- `-1`: New detections

---

**Detailed documentation in Polish**: [data/README.md](doc/README.md)
