from recognition.LLM import LLM_api

llm = LLM_api(model="deepseek-reasoner", max_workers=5)  # Create instance with 5 worker threads

# Single request (as before)
response = llm.get_text("Hello")

# Multiple parallel requests
contents = ["Hello", "How are you?", "What's the weather?"]
responses = llm.get_texts(contents)  # Returns list of responses