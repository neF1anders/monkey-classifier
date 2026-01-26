monkey-classifier/
│
├── docker/
│   ├── trainer.Dockerfile
│   ├── api.Dockerfile
│   └── streamlit.Dockerfile
│
├── data/
│   ├── raw/          # CSV only (tracked by git)
│   └── processed/    # images (tracked by DVC)
│
├── src/
│   ├── data/
│   │   └── download_from_inat.py
│   │   └── preprocess.py
│   ├── models/
│   ├── train.py
│   └── infer.py
│
├── api/
├── streamlit_app/
│
├── configs/
│   └── train.yaml
│
├── dvc.yaml
├── params.yaml
├── docker-compose.yml
└── README.md
