from flask import Flask, request, jsonify
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

app = Flask(__name__)

# Load the pre-trained model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

@app.route('/chat', methods=['POST'])
def chat():
    # Parse input JSON
    input_json = request.get_json()
    query = input_json['query']

    # Tokenize the input text
    inputs = tokenizer(query, return_tensors='pt')

    # Generate response from the model
    response = model.generate(**inputs)

    # Decode the response tokens to text
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)

    # Return response as JSON
    output = {'response': response_text}
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
