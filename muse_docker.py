from fastapi import FastAPI, HTTPException, Request
import tensorflow as tf

########## Serve the model by FastAPI ##########
model_g = tf.keras.models.load_model("./model/data/model")

app = FastAPI()
@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}

@app.post("/invocations")
async def invocations(request: Request):

    try:

        data = await request.json()  # Parses the raw JSON body
        eeg_l = data["inputs"] # [1, 256, 4]

        pred_l = model_g.predict(eeg_l)

        return {"predictions": pred_l.tolist()}

    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    pass