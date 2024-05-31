# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd

app = FastAPI()

class Item(BaseModel):
    DC: float
    DMC: float
    FFMC: float
    # Add more features as needed

@app.post("/predict")
async def predict(features: Item):
    try:
        # Load your pre-trained machine learning model
        model = joblib.load('model/base_model.pkl')
        preprocessor = joblib.load('model/preprocessor.joblib')

        # Prepare input features for prediction
        print(features.__dict__)
        features_df = pd.DataFrame(features.__dict__, index=[0])

        # Make predictions
        prediction = model.predict(features_df)

        # Return the prediction as JSON response
        return {"predicted class": prediction[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
