import torch
import numpy as np
import cv2
import librosa
import sounddevice as sd
from ultralytics import YOLO
from transformers import AutoFeatureExtractor, Wav2Vec2Model, AutoTokenizer, AutoModel
from scipy.stats import pearsonr
import threading
import time
import pandas as pd
from collections import deque

class MultimodalMLBugDetector:
    def __init__(self, sample_rate=16000, duration=5):
        # Visual: YOLO for object/lens detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Audio: VGGish-like (Wav2Vec2)
        self.audio_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Code/Input: CodeBERT
        self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")
        
        self.sample_rate = sample_rate
        self.duration = duration
        self.buffer = deque(maxlen=10)  # Store embeddings
        
        # Thresholds
        self.corr_thresh = 0.85
        self.corrs_detected = 0
        
    def get_visual_emb(self, frame):
        results = self.yolo(frame, verbose=False)
        # Dummy emb from detections (extend with CLIP/ResNet)
        confs = np.array([r.boxes.conf.cpu().numpy() for r in results if r.boxes is not None])
        return np.mean(confs) if len(confs) > 0 else np.random.rand(1)
    
    def get_audio_emb(self, audio):
        inputs = self.audio_extractor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def get_input_emb(self, input_data):  # e.g., code snippet or data types
        text = str(input_data)
        inputs = self.code_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.code_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def extract_types(self, data):
        types = {}
        for k, v in data.items():
            types[k] = str(type(v).__name__) if not torch.is_tensor(v) else f'tensor[{v.dtype}]'
        return types
    
    def find_cross_corrs(self, embs: List[np.ndarray], types: dict):
        """Cross-modal correlations on compatible types."""
        df = pd.DataFrame([emb for emb in embs])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        anomalies = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                t1, t2 = types.get('visual', 'float'), types.get('audio', 'float')  # Modal types
                if ('float' in t1 and 'float' in t2) or ('tensor' in t1 and 'float' in t2):  # Compatible
                    corr, _ = pearsonr(df[col1].dropna(), df[col2].dropna())
                    if abs(corr) > self.corr_thresh:
                        anomalies.append(f"corr({corr:.2f}): {col1}-{col2}")
        return anomalies
    
    def alert(self, anomalies):
        if len(anomalies) >= 2:
            print(f"🚨 MULTIMODAL BUG/SPY ALERT: {len(anomalies)} compatible correlations!")
            print("\n".join(anomalies))
            self.corrs_detected += 1
    
    def capture_stream(self):
        print("Starting multimodal capture (Ctrl+C to stop)...")
        while True:
            # Audio capture
            audio = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype=np.float32)
            sd.wait()
            audio_emb = self.get_audio_emb(audio.flatten())
            
            # Visual (webcam)
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                visual_emb = self.get_visual_emb(frame)
            else:
                visual_emb = np.random.rand()
            
            # Input data (demo: synthetic types)
            input_data = {'tensor': torch.tensor([np.random.rand()]), 'float': visual_emb}
            input_emb = self.get_input_emb(input_data)
            types = self.extract_types(input_data)
            
            embs = [visual_emb, audio_emb[:min(len(audio_emb), 768)], input_emb[:min(len(input_emb), 768)]]
            self.buffer.append(embs)
            
            if len(self.buffer) == self.buffer.maxlen:
                anomalies = self.find_cross_corrs([emb for sublist in list(self.buffer) for emb in sublist], types)
                self.alert(anomalies)
            
            time.sleep(0.1)

if __name__ == "__main__":
    detector = MultimodalMLBugDetector()
    try:
        detector.capture_stream()
    except KeyboardInterrupt:
        print("Stopped.")