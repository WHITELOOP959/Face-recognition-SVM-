# Face Recognition System Using SVM

Real-time face recognition using Support Vector Machines and HOG features.

---

## Overview

Recognizes faces through webcam using OpenCV Haar Cascade for detection and SVM with HOG features for classification.

**Accuracy:** 85-95% with proper training data.

---

## Features

- Load from dataset or capture via webcam
- Real-time recognition with confidence scores
- Save/load trained models
- Visualizations (confusion matrix, accuracy graphs)

---

## System Architecture

```
Input → Face Detection (Haar) → Preprocessing → HOG Features → SVM Classifier → Output
```

**Pipeline:**
1. **Input:** Webcam or dataset images
2. **Detection:** Haar Cascade finds faces
3. **Preprocessing:** Convert to grayscale, resize to 100×100
4. **Features:** Extract HOG (2916 dimensions)
5. **Classification:** SVM with RBF kernel predicts person
6. **Output:** Name + confidence score

---

## Key Components

### Configuration
```python
MODEL_PATH = 'face_svm_model.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
DATA_PATH = 'face_data.pkl'
IMG_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 0.6
IMAGES_PER_PERSON = 30
DATASET_PATH = r'C:\Users\user\Downloads\archive\Original Images\Original Images'
```

### HOG Feature Extraction
```python
def extract_hog_features(self, image):
    image = cv2.resize(image, IMG_SIZE)
    win_size = (100, 100)
    block_size = (20, 20)
    block_stride = (10, 10)
    cell_size = (10, 10)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog.compute(image).flatten()
```

**HOG** captures edge orientations, robust to lighting changes.

### Face Detection
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
```

### SVM Training
```python
def train(self):
    features = [self.extract_hog_features(face) for face in self.faces]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    self.model.fit(X_train, y_train)
```

**SVM Parameters:**
- `kernel='rbf'`: Non-linear classification
- `C=1.0`: Regularization strength
- `gamma='scale'`: Kernel coefficient
- `probability=True`: Enable confidence scores

### Recognition
```python
def recognize_face(self, face_roi):
    features = self.extract_hog_features(face_roi)
    probabilities = self.model.predict_proba([features])[0]
    max_prob = np.max(probabilities)
    if max_prob < CONFIDENCE_THRESHOLD:
        return "Unknown", max_prob
    predicted_label = self.model.predict([features])[0]
    person_name = self.label_encoder.inverse_transform([predicted_label])[0]
    return person_name, max_prob
```

### Real-time Recognition
```python
def start_webcam_recognition(self):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            name, confidence = self.recognize_face(face_roi)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```

---

## Usage

### Run the Application
```bash
cd C:\Users\user\OneDrive\Desktop\ML
python face.py
```

### Menu Options
**First Time (No Model):**
1. Load from dataset - Auto-loads images from archive folder
2. Capture from webcam - Manual capture of 30 images/person

**With Existing Model:**
1. Start recognition - Begin real-time detection
2. Retrain from dataset - Reload all images and retrain
3. Add people from webcam - Add new people to model
4. Clear all data - Delete model and start fresh

### Controls
- Press **'s'** to start capturing (webcam mode)
- Press **'q'** to quit recognition
- **Green box** = Recognized person
- **Red box** = Unknown person

---

## Mathematics

### HOG Features
1. **Gradients:** $G_x = I(x+1,y) - I(x-1,y)$, $G_y = I(x,y+1) - I(x,y-1)$
2. **Magnitude:** $M = \sqrt{G_x^2 + G_y^2}$
3. **Orientation:** $\theta = \arctan(G_y/G_x)$
4. **Histogram:** Bin orientations (0°-180°, 9 bins)
5. **Normalize:** L2-norm over blocks

### SVM Decision Function
$$f(\mathbf{x}) = \sum_{i} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$$

**RBF Kernel:**
$$K(\mathbf{x}_i, \mathbf{x}) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}||^2)$$

**Prediction:**
$$\hat{y} = \arg\max_k P(y=k|\mathbf{x})$$

---

## Output Files

| File | Description |
|------|-------------|
| `face_svm_model.pkl` | Trained SVM classifier |
| `label_encoder.pkl` | Name-to-number mapping |
| `face_data.pkl` | Cached face images and labels |
| `confusion_matrix.png` | Classification accuracy matrix |
| `training_results.png` | 4-panel performance graphs |

---

## Project Structure

```
ML/
├── face.py                    # Main application
├── face_svm_model.pkl        # Trained model
├── label_encoder.pkl         # Label encoder
├── face_data.pkl             # Face cache
├── confusion_matrix.png      # Confusion matrix
├── training_results.png      # Training graphs
└── README.md                 # Documentation
```

---

## Class Structure

```python
class FaceRecognitionSVM:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.face_cascade = cv2.CascadeClassifier(...)
        self.faces = []
        self.labels = []
    
    # Core Methods:
    extract_hog_features(image)           # Extract HOG features
    load_images_from_dataset()            # Load from folder
    capture_training_images()             # Webcam capture
    train()                               # Train SVM
    load_model()                          # Load saved model
    recognize_face(face_roi)              # Predict person
    start_webcam_recognition()            # Real-time loop
```

---

## Training Process

1. **Load Images:** From dataset folder or webcam
2. **Detect Faces:** Haar Cascade finds face regions
3. **Extract Features:** HOG features (2916 dims) from each face
4. **Encode Labels:** Convert names to integers
5. **Split Data:** 80% train, 20% test
6. **Train SVM:** Fit RBF kernel SVM
7. **Evaluate:** Calculate accuracy, confusion matrix
8. **Visualize:** Generate graphs
9. **Save:** Store model and encoder

---

## Performance Metrics

- **Accuracy:** Correct predictions / Total predictions
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Shows which people are misclassified

---

## Adjustable Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `IMG_SIZE` | (100, 100) | Larger = more accurate, slower |
| `CONFIDENCE_THRESHOLD` | 0.6 | Lower = more detections, more false positives |
| `IMAGES_PER_PERSON` | 30 | More = better accuracy (20-50 recommended) |
| `scaleFactor` | 1.1 | Lower = more accurate detection, slower |
| `minNeighbors` | 5 | Higher = fewer false detections |
| `C` | 1.0 | SVM regularization strength |
| `gamma` | 'scale' | RBF kernel coefficient |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low accuracy (<70%) | Capture 30-50 images per person, vary lighting/angles |
| "Unknown" for known faces | Lower `CONFIDENCE_THRESHOLD` to 0.5 |
| False positives | Increase `CONFIDENCE_THRESHOLD` to 0.7 |
| Webcam not opening | Close other camera apps, try `VideoCapture(1)` |
| Slow recognition | Reduce `IMG_SIZE` to (80, 80) |

---

## Best Practices

### Training Data
✅ **Do:**
- 30-50 images per person
- Vary lighting, angles, expressions
- Clear, unobstructed faces
- Good quality images

❌ **Don't:**
- Blurry or low-resolution images
- Multiple faces in one image
- Heavily edited/filtered photos

### Performance Tuning
- **Speed:** Smaller `IMG_SIZE`, higher `scaleFactor`
- **Accuracy:** More training images, lower `scaleFactor`, larger `IMG_SIZE`

---

## How It Works

**Training Flow:**
```
Dataset Images → Face Detection → Resize (100×100) → HOG Features → 
Label Encoding → Train/Test Split → SVM Training → Save Model
```

**Recognition Flow:**
```
Webcam Frame → Face Detection → Resize (100×100) → HOG Features → 
SVM Prediction → Confidence Check → Display Result
```

**Why SVM + HOG?**
- HOG captures face structure (edges, shapes)
- SVM finds optimal boundaries between faces
- Fast prediction (~30 FPS)
- Works with small datasets (30-50 images/person)

---

## Dependencies

```bash
pip install opencv-python numpy scikit-learn matplotlib seaborn joblib
```

**Required Libraries:**
- `opencv-python`: Face detection, image processing
- `numpy`: Array operations
- `scikit-learn`: SVM classifier, metrics
- `matplotlib`: Plotting
- `seaborn`: Enhanced visualizations
- `joblib`: Model serialization

---

## Technical Details

**HOG Dimensions:** 
- Window: 100×100
- Cells: 10×10 grid = 100 cells
- Blocks: 9×9 grid = 81 blocks
- Features per block: 4 cells × 9 bins = 36
- Total: 81 × 36 = 2916 features

**SVM Complexity:**
- Training: O(n² × d + n³)
- Prediction: O(n_sv × d)
- n = samples, d = features, n_sv = support vectors

---

**Version:** 1.0  
**Python:** 3.8+  
**Accuracy:** 85-95%  
**Last Updated:** December 5, 2025
