import openai
print("B")
response = openai.ChatCompletion.create(
    model="openai-3.5-turbo",
    messages=[
        #{"role": "system", "content": "You are a helpful tutor."},
        {"role": "user", "content": "Explain embeddings in simple words."}
    ]
)
print("A")
print(response['choices'][0]['message']['content'])
