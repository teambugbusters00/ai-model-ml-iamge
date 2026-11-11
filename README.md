# Image Classification API

FastAPI inference endpoint for a TorchScript image classifier.

```bash
pip install -r requirements.txt
MODEL_PATH=model.pt uvicorn main:app --reload
```

Export a compatible TorchScript model before calling POST /predict. The service returns a class index and confidence; add a label map and model-specific preprocessing before production use.