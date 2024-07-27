from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load model and tokenizer
model_path = 'model/'
model = GPT2LMHeadModel.from_pretrained(model_path, from_tf=False, use_safetensors=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    question = data.get('question', '')
    
    inputs = tokenizer(f"Question: {question}", return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
