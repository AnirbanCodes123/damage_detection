from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load the pre-trained YOLOv8 model
model = YOLO("trained.pt")

@app.post("/predict/")
async def predict(images: list[UploadFile] = File(...)):
    results = []
    for image in images:
        try:
            # Read the uploaded image
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))

            # Perform prediction
            predictions = model.predict(img, conf=0.25, imgsz=640)

            # Extract bounding box coordinates
            bbox_coordinates = []
            for prediction in predictions:
                boxes = prediction.boxes.xyxy.tolist()
                bbox_coordinates.extend(boxes)

            if not bbox_coordinates:
                results.append({"filename": image.filename, "status": "no detection"})
            else:
                results.append({
                    "filename": image.filename,
                    "bbox_coordinates": bbox_coordinates,
                    "status": "success"
                })

        except Exception as e:
            results.append({"filename": image.filename, "status": f"error: {str(e)}"})

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=443)
