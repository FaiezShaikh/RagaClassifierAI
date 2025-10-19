
import openai

# Load your Gemini API key
openai.api_key = "YOUR_API_KEY"

def classify_raga(features):
    prompt = open("prompts/raga_prompt.txt").read()
    user_input = f"Features: {features}"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response['choices'][0]['message']['content']
