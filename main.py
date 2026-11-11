"""Image inference API. Set MODEL_PATH to a TorchScript model file."""
import os,io
from fastapi import FastAPI,UploadFile,File,HTTPException
from PIL import Image
import torch
from torchvision import transforms
app=FastAPI(title="Image Classification API")
MODEL_PATH=os.getenv("MODEL_PATH","model.pt")
model=torch.jit.load(MODEL_PATH,map_location="cpu").eval() if os.path.exists(MODEL_PATH) else None
preprocess=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
@app.get("/health")
def health():return {"status":"ok","model_loaded":model is not None}
@app.post("/predict")
async def predict(file:UploadFile=File(...)):
 if model is None:raise HTTPException(503,"Model not found. Export a TorchScript model and set MODEL_PATH.")
 if file.content_type not in {"image/jpeg","image/png","image/webp"}:raise HTTPException(415,"Upload a JPEG, PNG or WEBP image.")
 try:img=Image.open(io.BytesIO(await file.read())).convert("RGB")
 except Exception:raise HTTPException(400,"Invalid image file.")
 with torch.inference_mode():scores=model(preprocess(img).unsqueeze(0))[0];idx=int(scores.argmax())
 return {"class_index":idx,"confidence":float(scores.softmax(0)[idx])}