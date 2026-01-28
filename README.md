# AI Image Classification Pipeline

End-to-end machine learning pipeline for image classification with experiment tracking, model serving, and vector search.

## 🚀 Features
- **Training Pipeline** with DVC for reproducibility
- **FastAPI** backend for model inference
- **Streamlit** dashboard for visualization
- **Qdrant** vector database for similarity search
- **Docker Compose** for containerized deployment


## 🛠️ Quick Start

### Prerequisites
- Docker & Docker Compose
- Git

### Local Development
```bash
# Clone repository
git clone git@github.com:neF1anders/monkey-classifier.git
cd <your dir>

# Run with Docker
docker compose build
docker compose up -d
docker compose exec trainer bash
streamlit run src/ui/streamlit_app.py --server.port=8502 --server.address=0.0.0.0

# Experiment with DVC
dvc exp run
dvc exp show

# Access services
# Jupyter: http://localhost:8080
# API: http://localhost:8000
# Streamlit: http://localhost:8502
# Qdrant: http://localhost:6333

🔗 Resources & References

    [DVC Documentation](https://doc.dvc.org/)

    [FastAPI Documentation](https://fastapi.tiangolo.com/)

    [Qdrant Documentation](https://qdrant.tech/documentation/)

    [Streamlit Documentation](https://docs.streamlit.io/)

    [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)

