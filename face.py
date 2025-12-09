import cv2
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configuration
MODEL_PATH = 'face_svm_model.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
DATA_PATH = 'face_data.pkl'
IMG_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 0.6
IMAGES_PER_PERSON = 30
DATASET_PATH = r'C:\Users\user\Downloads\archive\Original Images\Original Images'

class FaceRecognitionSVM:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.faces = []
        self.labels = []
    
    def extract_hog_features(self, image):
        """Extract HOG features from grayscale image"""
        image = cv2.resize(image, IMG_SIZE)
        
        # HOG parameters - adjusted to match 100x100 image size
        win_size = IMG_SIZE
        block_size = (20, 20)  # Changed to divide evenly
        block_stride = (10, 10)  # Changed to divide evenly
        cell_size = (10, 10)  # Changed to divide evenly
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(image)
        
        return features.flatten()
    
    def load_images_from_dataset(self):
        """Load face images from dataset folder"""
        print("\n=== LOADING IMAGES FROM DATASET ===")
        
        if not os.path.exists(DATASET_PATH):
            print(f"Error: Dataset path not found: {DATASET_PATH}")
            return False
        
        person_folders = [f for f in os.listdir(DATASET_PATH) 
                         if os.path.isdir(os.path.join(DATASET_PATH, f))]
        
        if not person_folders:
            print("No person folders found in dataset!")
            return False
        
        print(f"Found {len(person_folders)} people in dataset")
        print(f"People: {', '.join(person_folders)}\n")
        
        total_images = 0
        for person_name in person_folders:
            person_path = os.path.join(DATASET_PATH, person_name)
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Loading {person_name}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect face in image
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                )
                
                if len(faces) > 0:
                    # Take the largest face
                    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize face to standard size immediately
                    face_roi = cv2.resize(face_roi, IMG_SIZE)
                    
                    self.faces.append(face_roi)
                    self.labels.append(person_name)
                    total_images += 1
            
            print(f"  ✓ Loaded {len([l for l in self.labels if l == person_name])} faces for {person_name}")
        
        if len(self.faces) == 0:
            print("No faces detected in dataset!")
            return False
        
        # Save captured data
        data = {
            'faces': self.faces,
            'labels': self.labels
        }
        joblib.dump(data, DATA_PATH)
        print(f"\n✓ Total: {total_images} face images loaded and saved")
        
        return True
    
    def capture_training_images(self):
        """Capture face images from webcam for training"""
        print("\n=== CAPTURE TRAINING IMAGES ===")
        print("Instructions:")
        print("- Enter person's name")
        print("- Press 's' to start capturing")
        print("- 30 images will be captured automatically")
        print("- Move your head slightly for variety")
        print("- Type 'done' as name when finished\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        while True:
            person_name = input("Enter person's name (or 'done' to finish): ").strip()
            
            if person_name.lower() == 'done':
                break
            
            if not person_name:
                print("Please enter a valid name!")
                continue
            
            print(f"\nCapturing images for: {person_name}")
            print("Position your face in the frame and press 's' to start...")
            
            count = 0
            capturing = False
            
            while count < IMAGES_PER_PERSON:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                )
                
                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display instructions
                if not capturing:
                    cv2.putText(frame, "Press 's' to start capturing", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"Capturing: {count}/{IMAGES_PER_PERSON}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Move your head slightly", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('Capture Training Images', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Start capturing
                if key == ord('s') and not capturing:
                    capturing = True
                    print("Capturing started...")
                
                # Capture frames automatically
                if capturing and len(faces) > 0:
                    # Take the first detected face
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize face to standard size immediately
                    face_roi = cv2.resize(face_roi, IMG_SIZE)
                    
                    # Store face
                    self.faces.append(face_roi)
                    self.labels.append(person_name)
                    count += 1
                    
                    print(f"Captured {count}/{IMAGES_PER_PERSON}", end='\r')
                    time.sleep(0.1)  # Small delay between captures
                
                # Quit
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            print(f"\n✓ Captured {IMAGES_PER_PERSON} images for {person_name}")
            cv2.destroyAllWindows()
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(self.faces) == 0:
            print("No faces captured!")
            return False
        
        # Save captured data
        data = {
            'faces': self.faces,
            'labels': self.labels
        }
        joblib.dump(data, DATA_PATH)
        print(f"\n✓ Saved {len(self.faces)} face images for training")
        
        return True
    
    def load_captured_data(self):
        """Load previously captured face data"""
        if not os.path.exists(DATA_PATH):
            return False
        
        data = joblib.load(DATA_PATH)
        self.faces = data['faces']
        self.labels = data['labels']
        print(f"Loaded {len(self.faces)} previously captured faces")
        return True
    
    def train(self):
        """Train the SVM model"""
        if len(self.faces) == 0:
            print("No training data available!")
            return False
        
        print("\n=== TRAINING MODEL ===")
        
        # Ensure all faces are the same size
        faces_resized = []
        for face in self.faces:
            if face.shape != IMG_SIZE:
                face = cv2.resize(face, IMG_SIZE)
            faces_resized.append(face)
        
        faces = np.array(faces_resized)
        labels = np.array(self.labels)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print(f"Training with {len(faces)} images from {len(self.label_encoder.classes_)} people")
        print(f"People: {', '.join(self.label_encoder.classes_)}")
        
        # Extract features
        print("Extracting HOG features...")
        features = []
        for face in faces:
            hog_features = self.extract_hog_features(face)
            features.append(hog_features)
        
        features = np.array(features)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train SVM
        print("Training SVM classifier...")
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\n=== MODEL EVALUATION ===")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
        plt.close()
        
        # Training Accuracy Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Class Distribution
        unique, counts = np.unique(labels, return_counts=True)
        axes[0, 0].bar(unique, counts, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Training Data Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Person')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(counts):
            axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        # 2. Accuracy Metrics
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy
        metrics = ['Train Accuracy', 'Test Accuracy']
        values = [train_accuracy * 100, test_accuracy * 100]
        colors = ['#2ecc71', '#3498db']
        bars = axes[0, 1].bar(metrics, values, color=colors, edgecolor='black')
        axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].axhline(y=100, color='red', linestyle='--', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Per-Class Accuracy
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        x_pos = np.arange(len(self.label_encoder.classes_))
        width = 0.25
        axes[1, 0].bar(x_pos - width, precision * 100, width, label='Precision', color='#e74c3c')
        axes[1, 0].bar(x_pos, recall * 100, width, label='Recall', color='#f39c12')
        axes[1, 0].bar(x_pos + width, f1 * 100, width, label='F1-Score', color='#9b59b6')
        axes[1, 0].set_title('Per-Class Performance Metrics', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Person')
        axes[1, 0].set_ylabel('Score (%)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(self.label_encoder.classes_, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 105])
        
        # 4. Model Summary Info
        axes[1, 1].axis('off')
        summary_text = f"""
        MODEL SUMMARY
        {'='*40}
        
        Total Samples: {len(features)}
        Training Samples: {len(X_train)}
        Testing Samples: {len(X_test)}
        
        Number of Classes: {len(self.label_encoder.classes_)}
        Feature Vector Size: {features.shape[1]}
        
        SVM Kernel: RBF
        C Parameter: 1.0
        Gamma: scale
        
        Overall Accuracy: {accuracy * 100:.2f}%
        Train Accuracy: {train_accuracy * 100:.2f}%
        
        {'='*40}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5))
        
        plt.suptitle('SVM Face Recognition Training Results', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        print("✓ Training results graph saved as 'training_results.png'")
        plt.close()
        
        # Save model and encoder
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        print(f"✓ Model saved to '{MODEL_PATH}'")
        
        return True
    
    def load_model(self):
        """Load trained model and label encoder"""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
            return False
        
        self.model = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print("✓ Model loaded successfully!")
        print(f"Recognizes: {', '.join(self.label_encoder.classes_)}")
        return True
    
    def recognize_face(self, face_roi):
        """Recognize a face using trained model"""
        if self.model is None or self.label_encoder is None:
            return "Unknown", 0.0
        
        # Extract features
        features = self.extract_hog_features(face_roi)
        features = features.reshape(1, -1)
        
        # Predict with probability
        probabilities = self.model.predict_proba(features)[0]
        max_prob = np.max(probabilities)
        predicted_label = self.model.predict(features)[0]
        
        # Check confidence threshold
        if max_prob < CONFIDENCE_THRESHOLD:
            return "Unknown", max_prob
        
        person_name = self.label_encoder.inverse_transform([predicted_label])[0]
        return person_name, max_prob
    
    def start_webcam_recognition(self):
        """Real-time face recognition using webcam"""
        if self.model is None:
            print("Error: Model not loaded. Please train first.")
            return
        
        print("\n=== REAL-TIME FACE RECOGNITION ===")
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize face
                name, confidence = self.recognize_face(face_roi)
                
                # Draw rectangle
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label
                label = f"{name} ({confidence*100:.1f}%)"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display frame
            cv2.imshow('Face Recognition - Press q to quit', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Webcam closed")


def main():
    """Main execution function"""
    print("=" * 50)
    print("FACE RECOGNITION SYSTEM USING SVM")
    print("=" * 50)
    
    recognizer = FaceRecognitionSVM()
    
    # Check if model exists
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        print("\n✓ Found existing trained model")
        recognizer.load_model()
        
        choice = input("\nOptions:\n1. Start recognition\n2. Retrain from dataset\n3. Add people from webcam\n4. Clear all data and start fresh\nChoose (1/2/3/4): ").strip()
        
        if choice == '2':
            # Retrain from dataset
            if os.path.exists(DATA_PATH):
                os.remove(DATA_PATH)
            recognizer.load_images_from_dataset()
            recognizer.train()
        elif choice == '3':
            # Load existing data and add more from webcam
            recognizer.load_captured_data()
            recognizer.capture_training_images()
            recognizer.train()
        elif choice == '4':
            # Clear all data
            confirm = input("Are you sure you want to delete all trained data? (yes/no): ").strip().lower()
            if confirm == 'yes':
                for file_path in [MODEL_PATH, LABEL_ENCODER_PATH, DATA_PATH, 'confusion_matrix.png', 'training_results.png']:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"✓ Deleted {file_path}")
                print("\n✓ All data cleared! Restarting...\n")
                main()  # Restart
                return
            else:
                print("Cancelled.")
                return
        # else: choice == '1' or default, just start recognition
    else:
        print("\nNo trained model found. Let's create one!")
        
        choice = input("\nOptions:\n1. Load from dataset\n2. Capture from webcam\nChoose (1/2): ").strip()
        
        if choice == '1':
            # Load from dataset
            if not recognizer.load_images_from_dataset():
                print("Failed to load dataset.")
                return
        else:
            # Capture from webcam
            if not recognizer.capture_training_images():
                print("Training cancelled.")
                return
        
        # Train model
        if not recognizer.train():
            print("Training failed!")
            return
    
    # Start webcam recognition
    recognizer.start_webcam_recognition()


if __name__ == "__main__":
    main()
