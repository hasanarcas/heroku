from fastapi import FastAPI, UploadFile, File
import uvicorn
from prediction import read_image, preprocess, predict, encode
from starlette.responses import RedirectResponse

app = FastAPI()

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    image = read_image(await file.read())
    image = encode(image).reshape((1,2048))
    prediction = predict(image)
    print(prediction)
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='localhost')
    
