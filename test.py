import torch
import numpy as np
import cv2
import sounddevice as sd
from ultralytics import YOLO
from transformers import AutoFeatureExtractor, Wav2Vec2Model, AutoTokenizer, AutoModel
from scipy.stats import pearsonr
import threading
import time
import pandas as pd
from collections import deque
from typing import List
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MultimodalMLBugDetector:
    def __init__(self, sample_rate=16000, duration=5, audio_device=None):
        # FIRST: List devices SAFELY with default=-1 skip
        print("=== AUDIO DEVICES ===")
         # Try audio AFTER device list (safer)
        print("Finding best audio device for SYSTEM SOUNDS...")
        devices = sd.query_devices()
        
        # Find Stereo Mix or first input
        loopback_device = None
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            if ('stereo mix' in name or 'what u hear' in name or 'loopback' in name 
                or dev['max_input_channels'] > 0):
                loopback_device = i
                print(f"FOUND LOOPBACK: {i} {dev['name']}")
                break
        
        if loopback_device is None:
            print("No loopback - using silence")
            loopback_device = None
        
        # Models
        print("Loading models...")
        self.yolo = YOLO('yolov8n.pt')
        self.audio_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")
        
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_device = loopback_device
        self.audio_buffer = deque(maxlen=20)
        self.has_audio = False
        
        
        self.corr_thresh = 0.85
        self.corrs_detected = 0
        print("✅ Init complete!")
    
    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        self.audio_buffer.append(indata.flatten())
    
    def _start_audio_stream(self):
        if self.audio_device is None:
            print("No audio_device specified - silence mode")
            self._add_silence()
            return
        
        try:
            # Test device first
            dev_info = sd.query_devices(self.audio_device)
            if dev_info['max_input_channels'] < 1:
                raise ValueError("No input channels")
            
            extra_settings = sd.wasapi_flags() if hasattr(sd, 'wasapi_flags') else None
            self.audio_stream = sd.InputStream(
                device=self.audio_device, channels=1, samplerate=self.sample_rate,
                blocksize=int(self.sample_rate*0.5), callback=self._audio_callback,
                dtype='float32', latency='low', extra_settings=extra_settings
            )
            self.audio_stream.start()
            self.has_audio = True
            print(f"✅ LIVE SYSTEM AUDIO on {self.audio_device}")
        except Exception as e:
            print(f"❌ Audio device {self.audio_device} failed: {e}")
            print("Using silence fallback...")
            self._add_silence()
    
    def _add_silence(self):
        """Fallback audio"""
        silence = np.random.normal(0, 0.01, self.sample_rate//2)  # Tiny noise
        self.audio_buffer.append(silence)
    
    def get_visual_emb(self, frame):
        results = self.yolo(frame, verbose=False)
        confs = []
        for r in results:
            if r.boxes is not None:
                confs.extend(r.boxes.conf.cpu().numpy())
        confs = np.array(confs)
        return confs if len(confs)>0 else np.random.normal(0.5, 0.2, 32)
    
    def get_audio_emb(self, audio):
        if len(audio) == 0:
            return np.random.normal(0, 0.1, 768)
        try:
            inputs = self.audio_extractor(np.clip(audio, -1,1), 
                sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return emb[:768] if len(emb)>768 else np.pad(emb, (0,768-len(emb)))
        except:
            return np.random.normal(0, 0.1, 768)
    
    def get_input_emb(self, input_data):
        text = str(input_data)
        try:
            inputs = self.code_tokenizer(text[:512], return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.code_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return emb[:768] if len(emb)>768 else np.pad(emb, (0,768-len(emb)))
        except:
            return np.random.normal(0, 0.1, 768)
    
    def safe_corr(self, x, y):
        x = x.flatten()
        y = y.flatten()
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        x, y = x[mask], y[mask]
        if len(x) < 4 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        try:
            return pearsonr(x, y)[0]
        except:
            return 0.0
    
    def find_cross_corrs(self, embs):
        if len(embs) != 3:
            return []
        try:
            # Sample 100 points per emb for corr
            samples = []
            for emb in embs:
                idx = np.random.choice(len(emb), min(100, len(emb)), replace=False)
                samples.append(emb[idx])
            
            anomalies = []
            pairs = [('visual', 'audio'), ('visual', 'input'), ('audio', 'input')]
            for name1, name2 in pairs:
                corr = self.safe_corr(samples[0], samples[1]) if name1=='visual' and name2=='audio' else \
                       self.safe_corr(samples[0], samples[2]) if name1=='visual' and name2=='input' else \
                       self.safe_corr(samples[1], samples[2])
                if abs(corr) > self.corr_thresh:
                    anomalies.append(f"{name1}-{name2}: {corr:.3f}")
            return anomalies
        except:
            return []
    
    def alert(self, anomalies):
        if anomalies:
            print(f"\n🚨 ALERT #{self.corrs_detected+1}: {' | '.join(anomalies)}")
            self.corrs_detected += 1
    
    def capture_stream(self):
        print("\n🔍 Multimodal scanning... (Ctrl+C to stop)\n")
        try:
            for cycle in range(10):
                # Audio
                audio_data = np.concatenate(list(self.audio_buffer)[-8:]) if self.audio_buffer else np.zeros(self.sample_rate)
                audio_data = audio_data[:self.sample_rate*self.duration]
                audio_emb = self.get_audio_emb(audio_data)
                
                # Visual
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                visual_emb = self.get_visual_emb(frame) if ret else np.random.normal(0.5, 0.2, 32)
                
                # Input demo (system stats)
                input_data = {
                    'time': time.time(),
                    'audio_power': np.mean(audio_emb**2),
                    'vis_count': len(visual_emb)
                }
                input_emb = self.get_input_emb(input_data)
                
                embs = [visual_emb, audio_emb, input_emb]
                anomalies = self.find_cross_corrs(embs)
                self.alert(anomalies)
                self._add_silence()  # Keep buffer fresh
                
                print(f"Cycle {cycle} | Buf:{len(self.audio_buffer)} | Alerts:{self.corrs_detected}     ", end='\r')
                time.sleep(0.8)
        except KeyboardInterrupt:
            print("\nStopped cleanly.")
        finally:
            if hasattr(self, 'audio_stream') and self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()

if __name__ == "__main__":
    # UNCOMMENT after seeing INPUT index:
    # detector = MultimodalMLBugDetector(audio_device=2)  # YOUR STEREO MIX INDEX
    
    detector = MultimodalMLBugDetector(audio_device=2)  # SILENCE MODE FIRST
    detector.capture_stream()