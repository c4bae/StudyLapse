<div align="center">
  <h1>StudyLapse</h1>
</div> 

<p align="center">
  <kbd>
    <img width="1469" height="745" alt="Screenshot 2026-01-06 at 5 03 44 PM" src="https://github.com/user-attachments/assets/b02eae10-f28e-478e-a612-d34ddb0c6e4f" />
  </kbd>
</p>

<p align="center">
  <kbd>
    <img width="1460" height="745" alt="Screenshot 2026-01-06 at 9 18 31 PM" src="https://github.com/user-attachments/assets/6b0082d6-5d92-4d06-a8f9-0582aa217728" />
  </kbd>
</p>

<p align="center">
  <kbd>
    <img width="581" height="745" alt="Screenshot 2026-01-06 at 9 16 14 PM" src="https://github.com/user-attachments/assets/565cbcca-df0c-47f2-9d80-2c6e9d21893e" />
  </kbd>
</p>

Upload a timelapse of your study session and get AI-powered insights on focus time, absences, and phone usage — with an interactive video player that highlights exactly when each event occurred.

## Tech Stack

**Frontend:** React, TypeScript, Vite, Tailwind CSS, Framer Motion

**Backend:** FastAPI (Python), Express.js

**AI:** YOLOv8 (person detection), Custom PyTorch CNN (phone classification)

**Infrastructure:** Supabase (PostgreSQL), AWS S3, Clerk Auth

---

## How It Works

1. Upload a timelapse video
2. Express.js stores the video in S3 and triggers FastAPI
3. FastAPI samples every 8th frame, runs YOLOv8 for absence detection, then a custom CNN for phone usage
4. Results (timestamps + percentages) are saved to Supabase
5. The dashboard displays analytics with a color-coded video player (red = absent, yellow = phone)

---

## AI Models

| Model | Task | Details |
|-------|------|---------|
| YOLOv8n | Person detection | Pre-trained nano variant, <50ms/frame |
| StudyHabitClassifier | Phone vs. studying | Custom ResNet-based CNN, 83% validation accuracy |

Sampling every 8th frame reduces compute by **87.5%** without losing behavioral context.

---

## Features

- AI detection of focus, absence, and phone usage
- Interactive video player with timestamped highlights
- Analytics dashboard with per-class filtering
- Background video processing (non-blocking)
- Secure S3 storage with signed URLs
- Clerk authentication

---

## API

| Endpoint | Server | Purpose |
|----------|--------|---------|
| `POST /api/upload` | Express (3000) | Upload video to S3 |
| `POST /api/webhooks` | Express (3000) | Clerk user creation |
| `POST /api/media` | Express (3000) | Generate signed S3 URLs |
| `POST /api/process` | FastAPI (8000) | Trigger AI video analysis |

---

## Note

The Supabase schema and trained model (`timelapse_model.pth`) are not included. You'll need to set up your own Supabase database and train a model using `backend/train.py`.
