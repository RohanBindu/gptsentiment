from fastapi import FastAPI
import uvicorn
import tensorflow as tf 
from transformers import BertTokenizer
import numpy as np
from InputTexts import InputText
from starlette.responses import FileResponse 
from fastapi.responses import JSONResponse

app = FastAPI()

tickets_model = tf.keras.models.load_model('tickets_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
reverse_map = {0: 'neutral', 1: 'good', 2: 'bad'}
oglist = [0, 1, 2]

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(model, processed_data, classes=oglist):
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]


@app.get("/")
def read_root():
    return FileResponse('index.html')

@app.post("/predict/")
async def predict_service(input_data: InputText):
    data = input_data.dict()
    input_text = data['short_description']
    processed_data = prepare_data(input_text, tokenizer)  # Replace 'tokenizer' with your tokenizer object
    result = make_prediction(tickets_model, processed_data=processed_data)  # Replace 'tickets_model' with your model
    predicted_service = reverse_map.get(result, "Service not found")  # Ensure a default value in case of missing result
    return predicted_service
  

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn main:app --reload

