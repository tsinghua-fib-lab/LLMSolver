import requests
import time
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, AzureOpenAI
from llm_config import api_config_dict, get_model_config, get_api_key, get_model_type
import copy

class OverloadError(Exception):
    pass

class LLM_api:
    def __init__(self, 
                 content="",
                 model="Qwen/Qwen2.5-7B-Instruct",
                 max_tokens=4096,
                 stop=None,
                 temperature=1,
                 top_p=0.7,
                 frequency_penalty=0,
                 n=1):

        self.payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "n": n,
            "response_format": {"type": "text"},
            "tools": []
        }
        
        if stop:
            self.payload["stop"] = stop  # Only add stop if not None
        
        # Get model configuration
        model_config = get_model_config(model)
        model_type = get_model_type(model)
        
        # Initialize client based on model type
        if model_type == "azure":
            print("Using Azure OpenAI API")
            self.HTTP_CLIENT = httpx.Client(proxy=model_config.get("proxy", "http://127.0.0.1:8456"))
            self.client = AzureOpenAI(
                api_key=model_config.get("api_key", ""),
                api_version=model_config.get("api_version", "2024-07-01-preview"),
                azure_endpoint=model_config.get("azure_endpoint", ""),
                http_client=self.HTTP_CLIENT
            )
        elif model_type == "openai":
            # Get API key
            api_key = get_api_key(model)
            if not api_key:
                raise ValueError(f"Missing API key for model {model}")
            
            self.client = OpenAI(
                base_url=model_config.get("base_url", "https://api.openai.com"),
                api_key=api_key
            )
        elif model_type == "siliconflow":
            # SiliconFlow API handling
            api_key = get_api_key(model)
            if not api_key:
                raise ValueError(f"Missing API key for model {model}")
            
            self.url = model_config.get("base_url", "https://api.siliconflow.cn/v1/chat/completions")
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.q_token = 0
        self.a_token = 0

    def reset_token(self):
        self.q_token = 0
        self.a_token = 0

    def get_token(self):
        return self.q_token, self.a_token

    def get_response(self, payload):
        """Handles API requests with retry mechanism"""
        max_try = 10
        sleep_time = 5
        max_sleep = 20

        for try_num in range(max_try):
            try:
                model = payload["model"]
                model_type = get_model_type(model)
                
                if model_type == "azure":
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=payload["messages"],
                        max_tokens=payload["max_tokens"],
                        temperature=payload["temperature"],
                        top_p=payload["top_p"],
                        frequency_penalty=payload["frequency_penalty"],
                        n=payload["n"],
                        timeout=60
                    )
                    content = response.choices[0].message.content
                    self.q_token += response.usage.prompt_tokens
                    self.a_token += response.usage.completion_tokens
                    if content:
                        return {"choices": [{"message": {"content": content}}]}
                
                elif model_type == "openai":
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=payload["messages"],
                        max_tokens=payload["max_tokens"],
                        temperature=payload["temperature"],
                        top_p=payload["top_p"],
                        frequency_penalty=payload["frequency_penalty"],
                        n=payload["n"],
                        timeout=60
                    )
                    content = response.choices[0].message.content
                    self.q_token += response.usage.prompt_tokens
                    self.a_token += response.usage.completion_tokens
                    
                    # Special handling for deepseek-reasoner reasoning_content
                    if model == "deepseek-reasoner":
                        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
                        if content:
                            if reasoning_content:
                                return {"choices": [{"message": {"content": content, "reasoning_content": reasoning_content}}]}
                            else:
                                return {"choices": [{"message": {"content": content}}]}
                    else:
                        if content:
                            return {"choices": [{"message": {"content": content}}]}
                            
                elif model_type == "siliconflow":
                    response = requests.request("POST", self.url, json=payload, headers=self.headers)
                    response = response.json()
                    content = response["choices"][0]["message"]["content"]
                    self.q_token += response["usage"]["prompt_tokens"]
                    self.a_token += response["usage"]["completion_tokens"]
                    if content:
                        return {"choices": [{"message": {"content": content}}]}
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

            except requests.RequestException as e:
                print(f"API request failed (attempt {try_num + 1}/{max_try}): {e}")
            except KeyError as e:
                print(f"Unexpected API response structure: {e}")
            except Exception as e:
                print(f"Unexpected error (attempt {try_num + 1}/{max_try}): {e}")

            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 2, max_sleep)

        print("Warning: TOO MANY TRIES, EMPTY RESPONSE")
        return {"choices": [{"message": {"content": ""}}]}
    
    def get_text(self, content=""):
        """Gets text response from LLM API"""
        payload_copy = copy.deepcopy(self.payload)
        payload_copy["messages"] = [{"role": "user", "content": content}]
        response = self.get_response(payload_copy)
        return response.get('choices', [{}])[0].get('message', {}).get('content', "")
    
    def get_multi_text(self, contents=[""], max_workers=5):
        """
        Handles multiple API requests concurrently using threads.
        
        Args:
            contents (list): A list of strings, each representing content to send to the API.
        
        Returns:
            list: A list of responses corresponding to the input contents.
        """
        results_with_index = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.get_text, content): i for i, content in enumerate(contents)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results_with_index[index] = result
                except Exception as e:
                    print(f"Error processing content '{contents[index]}': {e}")
                    results_with_index[index] = ""
        
        return [results_with_index[i] for i in range(len(contents))]

    def extract_state(self, res_string):
        """Extracts predefined scores from the response string"""
        state_key = ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "C1"]
        state = {}

        for key in state_key:
            location = res_string.find(key)
            if location != -1 and location + 4 < len(res_string):
                char = res_string[location + 4]
                state[key] = int(char) if char.isdigit() else -1
            else:
                state[key] = -1

        return state
    
    def set_content(self, content):
        """Updates the user query in the request payload"""
        self.payload["messages"][0]["content"] = content

    def print_usage(self):
        """Prints API token usage"""
        print(f"Prompt tokens: {self.q_token}")
        print(f"Completion tokens: {self.a_token}")


if __name__ == "__main__":
    llm = LLM_api(model="deepseek-chat")
    # Single thread call example
    text = llm.get_text(content="Hello")
    print("Single thread result:", text)

    # Multi-thread call example
    contents = ["你好", "こんにちは", "Hello", "안녕하세요", "Bonjour", "Hola", "Ciao", "Привет", "مرحبا", "Hello"]
    responses = llm.get_multi_text(contents)
    print("Multi-thread results:")
    for i, response in enumerate(responses):
        print(f"Content {contents[i]} response: {response}")