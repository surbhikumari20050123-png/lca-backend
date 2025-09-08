from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd
from io import StringIO

app = FastAPI(title="AI-Powered LCA Tool")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5180", "http://127.0.0.1:5173", "http://127.0.0.1:5174", "http://127.0.0.1:5175", "http://127.0.0.1:5180"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model_path = "lca_model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Global dataset variable
dataset_path = "dataset_expanded.csv"
if os.path.exists(dataset_path):
    dataset_df = pd.read_csv(dataset_path)
else:
    dataset_df = pd.DataFrame()

from fastapi import UploadFile, File, HTTPException

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and Excel files are supported.")
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            new_data = pd.read_csv(StringIO(contents.decode("utf-8")))
        else:
            import io
            new_data = pd.read_excel(io.BytesIO(contents))
        global dataset_df
        # Simple merge: append new data
        dataset_df = pd.concat([dataset_df, new_data], ignore_index=True)
        # Save updated dataset
        dataset_df.to_csv(dataset_path, index=False)
        return {"message": "Dataset uploaded and merged successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

from pydantic import BaseModel, validator

class InputData(BaseModel):
    energy_use: float = None
    recycled_content: float
    material_type: str
    route: str
    energy_source: str
    transport_mode: str
    end_of_life: str
    quantity: float
    distance: float

    @validator('energy_use', pre=True, always=True)
    def set_energy_use_default(cls, v):
        # Allow energy_use to be optional and default to None
        return v if v is not None else 0.0

@app.options("/predict")
def options_predict():
    return {"message": "OK"}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Model not trained yet. Run train_model.py first."}

    # Debug logging
    print(f"Received data: material_type={data.material_type}, route={data.route}, energy_source={data.energy_source}, transport_mode={data.transport_mode}")

    # Preprocess input data to match training data format
    import pandas as pd

    # Create input DataFrame with all features
    input_data = pd.DataFrame({
        "Recycled Content (%)": [data.recycled_content],
        "Distance (km)": [data.distance],
        "Quantity (kg)": [data.quantity],
        "Material Type": [data.material_type],
        "Route": ["Raw Material" if data.route == "raw" else "Recycled"],
        "Energy Source": [data.energy_source.capitalize()],
        "Transport Mode": ["Road" if data.transport_mode == "truck" else data.transport_mode.capitalize()],
        "End-of-Life Option": [data.end_of_life.capitalize()]
    })

    # Apply one-hot encoding to categorical columns (same as training)
    categorical_cols = ["Material Type", "Route", "Energy Source", "Transport Mode", "End-of-Life Option"]
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols)

    # Ensure all columns from training are present (add missing columns with 0)
    expected_columns = [
        "Recycled Content (%)", "Distance (km)", "Quantity (kg)",
        "Material Type_Aluminium", "Material Type_Chromium", "Material Type_Copper",
        "Material Type_Lead", "Material Type_Magnesium", "Material Type_Manganese",
        "Material Type_Nickel", "Material Type_Steel", "Material Type_Titanium", "Material Type_Zinc",
        "Route_Raw Material", "Route_Recycled",
        "Energy Source_Coal", "Energy Source_Gas", "Energy Source_Hydro", "Energy Source_Solar", "Energy Source_Wind",
        "Transport Mode_Air", "Transport Mode_Rail", "Transport Mode_Road", "Transport Mode_Ship",
        "End-of-Life Option_Incinerate", "End-of-Life Option_Landfill", "End-of-Life Option_Recycle", "End-of-Life Option_Reuse"
    ]

    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match training data
    input_encoded = input_encoded[expected_columns]

    # Make prediction using trained model
    predicted_energy_use = model.predict(input_encoded)[0]
    print(f"Predicted energy use: {predicted_energy_use}")

    # Calculate CO2 emissions based on predicted energy use and other factors
    co2_emissions = predicted_energy_use * 0.5  # Convert energy use to emissions
    print(f"CO2 emissions: {co2_emissions}")

    # Additional adjustments for CO2
    energy_factors = {"coal": 1.5, "hydro": 0.5, "solar": 0.1, "wind": 0.2, "gas": 1.2}
    co2_emissions *= energy_factors.get(data.energy_source, 1.0)

    transport_factors = {"truck": 1.2, "ship": 0.8, "rail": 0.6, "air": 2.0, "road": 1.1}
    co2_emissions *= transport_factors.get(data.transport_mode, 1.0)

    co2_emissions += data.distance * 0.01
    co2_emissions *= data.quantity / 1000

    # Calculate SOx emissions (sulfur oxides)
    sox_base = predicted_energy_use * 0.004  # Base SOx emission factor
    sox_factors = {"coal": 2.0, "gas": 1.5, "hydro": 0.1, "solar": 0.05, "wind": 0.08}
    sox_emissions = sox_base * sox_factors.get(data.energy_source, 1.0) * (data.quantity / 1000)

    # Calculate NOx emissions (nitrogen oxides)
    nox_base = predicted_energy_use * 0.006  # Base NOx emission factor
    nox_factors = {"coal": 1.8, "gas": 1.6, "hydro": 0.2, "solar": 0.1, "wind": 0.15}
    nox_emissions = nox_base * nox_factors.get(data.energy_source, 1.0) * (data.quantity / 1000)

    # Calculate water use
    water_base = predicted_energy_use * 50  # Base water use factor (L/kWh)
    water_factors = {"coal": 2.5, "gas": 1.8, "hydro": 0.5, "solar": 0.1, "wind": 0.2}
    water_use = water_base * water_factors.get(data.energy_source, 1.0) * (data.quantity / 1000)

    # Calculate energy intensity
    energy_intensity = predicted_energy_use / data.quantity

    # Circularity indicators
    recycled_content_pct = data.recycled_content * 100
    resource_efficiency = 100 - (predicted_energy_use / 50 * 100)  # Higher is better
    reuse_recycling_potential = data.recycled_content * 100
    if data.end_of_life == "recycle":
        reuse_recycling_potential += 30
    elif data.end_of_life == "reuse":
        reuse_recycling_potential += 50

    # Overall circularity score
    circularity_score = (recycled_content_pct + resource_efficiency + reuse_recycling_potential) / 3

    # Recommendation
    recommendation = "Increase recycling to improve circularity" if data.recycled_content < 0.5 else "System is circular enough"
    if data.energy_source in ["solar", "wind"]:
        recommendation += " Good choice of renewable energy."
    if data.transport_mode == "rail":
        recommendation += " Efficient transport mode."

    return {
        "predicted_emissions": round(float(co2_emissions), 2),
        "sox_emissions": round(float(sox_emissions), 3),
        "nox_emissions": round(float(nox_emissions), 3),
        "water_use": round(float(water_use), 1),
        "energy_intensity": round(float(energy_intensity), 3),
        "circularity_score": round(float(circularity_score), 2),
        "recycled_content_pct": round(float(recycled_content_pct), 1),
        "resource_efficiency": round(float(resource_efficiency), 1),
        "reuse_recycling_potential": round(float(reuse_recycling_potential), 1),
        "recommendation": recommendation
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
