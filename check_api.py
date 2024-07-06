import openai

def check_openai_key(api_key):
    openai.api_key = api_key
    try:
        # Attempt to list available models to check the API key
        models_response = openai.Model.list()
        if models_response['object'] == 'list':
            print("API key is working for listing models.")
        else:
            print("Unexpected response format when listing models.")
        
        # Attempt to prompt GPT-4 model with a simple query
        completion_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Can you confirm if this API key is working?"}
            ]
        )
        
        print("GPT-4 response:")
        print(completion_response.choices[0].message['content'])
        
    except openai.error.AuthenticationError:
        print("Invalid API key.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'your-openai-api-key' with your actual OpenAI API key
api_key = 'sk-proj-IoxzbELHRwIIhrlZVwrtT3BlbkFJvyxGl7jRv3fEzURZJt6g'
check_openai_key(api_key)
