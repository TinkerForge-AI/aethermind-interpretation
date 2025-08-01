# annotators/vision_tags.py

import torch
import clip
import functools
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_candidate_tags(path=None):
    if path is None:
        path = Path(__file__).parent / "tag_vocab.txt"
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
# List of possible labels to match against — customize this list
CANDIDATE_TAGS = load_candidate_tags()

@torch.no_grad()
def tag_visual_scene(video_path, top_k=5):
    model, preprocess = get_clip_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract frames (you already do this)
    frames = extract_sample_frames(video_path)  # 120 frames
    candidate_tags = load_candidate_tags()

    # Encode all candidate tags
    with torch.no_grad():
        text_tokens = clip.tokenize(candidate_tags).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    image_features = []
    for frame in frames:
        img = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            image_features.append(feat)

    image_features = torch.cat(image_features, dim=0)  # (N, D)
    mean_feat = image_features.mean(dim=0, keepdim=True)

    similarities = (mean_feat @ text_features.T).squeeze(0)  # shape: (num_tags,)
    top_scores, top_idxs = similarities.topk(top_k)

    results = [
        {"label": candidate_tags[i], "score": round(score.item(), 4)}
        for i, score in zip(top_idxs, top_scores)
    ]

    return results

def log_unseen_tags(tag_dicts, vocab_path="tag_vocab.txt", log_path="tag_log.txt"):
    base_dir = Path(__file__).parent
    vocab_path = Path(vocab_path)
    if not vocab_path.is_absolute():
        vocab_path = base_dir / vocab_path
    log_path = Path(log_path)
    if not log_path.is_absolute():
        log_path = base_dir / log_path

    known = set(load_candidate_tags(vocab_path))
    new_tags = [t["label"] for t in tag_dicts if t["label"] not in known]
    if new_tags:
        with open(log_path, "a") as f:
            for t in new_tags:
                f.write(t + "\n")

@functools.lru_cache(maxsize=1)
def get_clip_model():
    model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    return model, preprocess

def extract_sample_frames(video_path, num_frames=120):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))

    cap.release()
    return frames