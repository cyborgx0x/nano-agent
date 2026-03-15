# 🎥 Testing SLAM with YouTube Videos

## Quick Start

### **Option 1: Use YouTube URL directly**

```bash
cd /home/lucas/Documents/ai-research/nano-agent
source .venv/bin/activate

# Find a good Albion Online gathering video on YouTube
# Example: "Albion Online fiber gathering gameplay"

python -m slam.test_slam_video "https://youtube.com/watch?v=VIDEO_ID"
```

### **Option 2: Use downloaded video file**

```bash
# If you already have a video file
python -m slam.test_slam_video albion_gameplay.mp4
```

---

## 📹 Recommended Test Videos

Search YouTube for:
- **"Albion Online gathering gameplay"**
- **"Albion Online farming fiber"**
- **"Albion Online black zone gathering"**
- **"Albion Online resource gathering bot"** (to see what competitors do!)

**What to look for:**
- ✅ **1080p or higher** (better feature detection)
- ✅ **5-10 minutes long** (enough to test drift)
- ✅ **Player moving around** (not standing still)
- ✅ **Visible minimap** (optional, for validation)
- ❌ Avoid night scenes (too dark for ORB features)
- ❌ Avoid PvP videos (too chaotic)

---

## 🎯 What the Test Shows

### **Real-time Display:**
```
Frame 100/3000 | Pos: (2543, 1832) | Objects: 3 | Time: 15.2ms | Avg: 14.8ms
```

- **Frame**: Current frame number
- **Pos**: Estimated position from SLAM
- **Objects**: Fiber nodes detected (if model loaded)
- **Time**: Processing time per frame
- **Avg**: Rolling average (last 10 frames)

### **Visual Overlay:**
- 🟢 Green text: Position and zone info
- 🟡 Yellow trail: Recent trajectory (last 50 positions)
- 🔴 Red dot: Current position

---

## 📊 Results Generated

After processing, you'll get:

### **1. Performance Graphs** (`slam/slam_video_results.png`)
Four plots showing:
- **Trajectory**: 2D path the player took
- **Distance**: Total distance traveled over time
- **Processing Time**: How fast SLAM runs (should be <20ms)
- **Objects Detected**: Resources found per frame

### **2. Discovered Map** (`discovered_map.png`)
Visual map showing:
- Explored areas (bright)
- Unexplored areas (black)
- Path traveled

### **3. SLAM Data** (`slam_maps/video_test_final.pkl`)
- Full SLAM state (can load later)
- Object database
- Zone graph
- Trajectory history

---

## 🧪 Validation Tests

### **Test 1: Does position track correctly?**

**Method:**
1. Note a distinctive landmark at start (e.g., big tree)
2. Player moves away for 2 minutes
3. Player returns to same tree
4. Check if SLAM position returns to ~same coordinates

**Expected:**
- Position error < 200 pixels after 2 min loop

**If fails:**
- Adjust `movement_scale` in `slam/slam_navigator.py`

---

### **Test 2: Does drift accumulate?**

**Method:**
1. Note starting position at frame 0
2. Let video play for 1000 frames
3. Check position change

**Expected:**
- Drift rate: <5 pixels/second of video

**If fails:**
- Need loop closure detection
- Or better feature matching (use SIFT instead of ORB)

---

### **Test 3: Are objects detected correctly?**

**Method:**
1. Pause video when you see fiber on screen
2. Check if "Objects: X" count matches
3. Look at saved map for object markers

**Expected:**
- 80%+ detection rate (if fiber model loaded)

**If fails:**
- Model might need retraining
- Or video resolution too low

---

## 🔧 Tuning Parameters

If SLAM doesn't work well, edit `slam/slam_navigator.py`:

```python
class SLAMNavigator:
    def __init__(self, ...):
        # TUNE THESE:

        # If position moves too fast
        self.movement_scale = 1.0  # Try: 0.5, 1.5, 2.0

        # If not enough features detected (dark video)
        self.orb = cv2.ORB_create(nfeatures=2000)  # Default: 1000

        # If matching fails often
        # Use SIFT (slower but more robust)
        self.orb = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
```

---

## 📈 Expected Results

**Good SLAM performance:**
```
✅ Processing speed: 10-20ms per frame
✅ Position drift: <5 px/sec
✅ Object detection: 80%+ accuracy
✅ Zones discovered: Matches video (1-3 zones)
✅ Trajectory looks smooth (no jumps)
```

**Bad SLAM performance (needs tuning):**
```
❌ Processing speed: >50ms (too slow)
❌ Position drift: >20 px/sec (accumulates fast)
❌ Object detection: <50% (model issues)
❌ Trajectory has jumps (lost tracking)
```

---

## 🐛 Troubleshooting

### **"Could not open video"**
```bash
# Check video file exists
ls -lh albion_gameplay.mp4

# Try different video format
ffmpeg -i input.mkv -c copy output.mp4
```

### **"Processing too slow (>50ms)"**
```python
# Use fewer ORB features
self.orb = cv2.ORB_create(nfeatures=500)  # Instead of 1000

# Skip frames
if frame_id % 2 == 0:  # Process every 2nd frame
    slam.update_map(frame)
```

### **"Position drifts rapidly"**
```python
# Reduce movement scale
self.movement_scale = 0.5

# Or use better features
self.orb = cv2.SIFT_create()  # More stable than ORB
```

### **"No features detected"**
```python
# Increase contrast
gray = cv2.equalizeHist(gray)

# Or use CLAHE (better)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
```

---

## 🎯 Next Steps After Testing

### **If SLAM works well (< 5% drift):**
1. ✅ Test on real gameplay (run while playing)
2. ✅ Add loop closure detection
3. ✅ Collect portal/landmark dataset
4. ✅ Build multi-zone navigation

### **If SLAM has issues:**
1. ❌ Calibrate `movement_scale` with manual annotations
2. ❌ Switch to SIFT features (more robust)
3. ❌ Add IMU simulation (predict movement from animations)
4. ❌ Implement Kalman filter (fuse multiple estimates)

---

## 📝 Manual Calibration

If automatic calibration doesn't work:

```python
# 1. Load video
cap = cv2.VideoCapture('test_video.mp4')

# 2. Mark known positions manually
ground_truth = [
    (0, 0, 0),        # frame 0: start at (0, 0)
    (300, 150, 200),  # frame 300: moved to (150, 200) on minimap
    (600, 300, 400),  # frame 600: moved to (300, 400)
]

# 3. Run SLAM and compare
for frame_id, true_x, true_y in ground_truth:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    # Process up to this frame
    slam.update_map(frame)

    # Compare
    est_x, est_y = slam.current_position
    error = np.sqrt((est_x - true_x)**2 + (est_y - true_y)**2)
    print(f"Frame {frame_id}: Error = {error:.0f} pixels")

# 4. Adjust movement_scale to minimize error
```

---

## 🏆 Success Criteria

Your SLAM is ready for production if:

✅ **Performance**: <20ms per frame
✅ **Accuracy**: <100px error after 5 minutes
✅ **Robustness**: Works on 3+ different videos
✅ **Objects**: Detects 70%+ of visible resources
✅ **Zones**: Correctly detects zone transitions

Then you can use it for:
- Multi-zone pathfinding
- Resource mapping
- Automated farming routes

---

## 💡 Pro Tips

1. **Test on multiple videos** - different zones, lighting, times of day
2. **Start with short clips** - 1-2 minutes first, then longer
3. **Compare trajectories** - manually trace path, compare with SLAM estimate
4. **Use minimap as ground truth** - if video shows minimap, validate against it
5. **Save intermediate results** - `save_interval=50` to see progression

---

## 🎬 Example Workflow

```bash
# 1. Download a test video
python -m slam.test_slam_video "https://youtube.com/watch?v=ALBION_VIDEO"

# 2. Process and analyze
# (Press 'q' to stop early if it's working)

# 3. Check results
open slam/slam_video_results.png
open discovered_map.png

# 4. Load and query the map
python -c "
from slam import SLAMNavigator
slam = SLAMNavigator()
slam.load_map('slam_maps/video_test_final.pkl')
print(f'Zones discovered: {len(slam.zones)}')
print(f'Objects found: {sum(len(v) for v in slam.object_database.values())}')
"

# 5. If good, test on real gameplay!
python -m slam.slam_navigator
```

---

Good luck! 🚀

**Remember**: SLAM is hard! Even 10% drift is acceptable for game navigation. We just need "good enough" to find resources and plan routes.
