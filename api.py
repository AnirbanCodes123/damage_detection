from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import json

app = FastAPI()

# Load the pre-trained YOLOv8 model
model = YOLO("trained.pt")

@app.post("/predict_damage/")
async def predict(image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        # Perform prediction
        results = model.predict(img, conf=0.25, device="mps", imgsz=640)

        # Extract bounding box coordinates
        bbox_coordinates = []
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            bbox_coordinates.extend(boxes)

        if not bbox_coordinates:
            return {"status": "no detection"}

        return {
            "bbox_coordinates": bbox_coordinates,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
