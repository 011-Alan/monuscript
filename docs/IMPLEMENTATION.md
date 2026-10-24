# MonuScript: Implementation Architecture & Technical Report

## 1. System Architecture Overview

```
                      +-----------------------------+
                      |   React + TypeScript UI     |
                      |   (Vite, Tailwind, Lucide)  |
                      +--------------+--------------+
                                     |  REST API (/api/v1)
                                     v
                      +-----------------------------+
                      |      Flask Backend API      |
                      +--------------+--------------+
                                     |
       +-----------------------------+-----------------------------+
       |                             |                             |
       v                             v                             v
+---------------+           +-----------------+           +-----------------+
| Preprocessing |           |   Deterministic |           | SQLite / PG     |
| & Safety      |           |   DIP Features  |           | Persistence     |
| - EXIF Orient |           |   - Grayscale   |           | - Conversations |
| - 256x256 Norm|           |     Hist (32)   |           | - Messages      |
| - Grayscale   |           |   - Stats (8)   |           | - Recognition   |
|   Formula     |           |   - Sobel (36)  |           |   Snapshots     |
| - HSV Color   |           |   - Texture (31)|           +-----------------+
+-------+-------+           |   - HSV (32)    |
        |                   +--------+--------+
        v                            |
+------------------------------------+-----------------------------+
|               StandardScaler (Trained on Train Split)            |
+------------------------------------+-----------------------------+
                                     |
                                     v
+------------------------------------------------------------------+
|                   Support Vector Machine (SVM)                   |
|                   - Linear / RBF Kernel Tuning                   |
|                   - Platt Calibrated Probabilities               |
+------------------------------------+-----------------------------+
                                     |
                                     v
+------------------------------------------------------------------+
|                   Multi-Criteria Rejection Judge                 |
|                   - Minimum Confidence (tau_accept = 0.40)       |
|                   - Runner-Up Margin (tau_margin = 0.12)         |
|                   - Feature Centroid Distance (OOD Detection)    |
+------------------------------------+-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Structured Result      |
                      | (Cards, 3D Links, Metadata) |
                      +-----------------------------+
```

---

## 2. Directory Layout
```
monuscript/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── ChatHeader.tsx
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── ImageUploader.tsx
│   │   │   ├── RecognitionResult.tsx
│   │   │   ├── MonumentCard.tsx
│   │   │   ├── ConfidenceIndicator.tsx
│   │   │   ├── EmptyState.tsx
│   │   │   ├── FeatureInspectionModal.tsx
│   │   │   ├── ThreeDModal.tsx
│   │   │   └── SupportedMonumentsModal.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── database/
│   │   ├── db.py
│   │   ├── schema.sql
│   │   └── supabase_schema.sql
│   ├── image_processing/
│   │   ├── preprocessing.py
│   │   ├── transformations.py
│   │   ├── histogram_features.py
│   │   ├── edge_features.py
│   │   ├── texture_features.py
│   │   └── feature_extractor.py
│   ├── ml/
│   │   ├── prepare_dataset.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   ├── inference.py
│   │   ├── rejection.py
│   │   └── model_artifacts/
│   │       ├── svm_pipeline.joblib
│   │       ├── model_metadata.json
│   │       └── evaluation_results.json
│   ├── services/
│   │   ├── recognition_service.py
│   │   └── monument_service.py
│   ├── data/
│   │   ├── monuments.json
│   │   ├── dataset_catalog.json
│   │   └── dataset/
│   │       ├── train/
│   │       ├── val/
│   │       ├── test/
│   │       └── negatives/
│   └── tests/
│       ├── test_preprocessing.py
│       ├── test_features.py
│       ├── test_inference.py
│       └── test_api.py
├── docs/
│   ├── FORMULAS.md
│   ├── DATASET.md
│   ├── TESTING.md
│   ├── LIMITATIONS.md
│   └── IMPLEMENTATION.md
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## 3. Database Persistence
Local storage uses SQLite (`backend/data/monuscript.db`) initialized automatically upon server startup.
* Tables: `conversations`, `messages`, `recognition_results`.
* Cascade deletes: Deleting a conversation removes all child messages and recognition snapshots.
* Production migration: `backend/database/supabase_schema.sql` provides full PostgreSQL DDL with Row Level Security (RLS) policies.
