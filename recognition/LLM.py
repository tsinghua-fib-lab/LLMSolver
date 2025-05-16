import requests
import time
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, AzureOpenAI
from recognition.API_key import api_key_dict
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
                 n=1,
                 key_idx=0):

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
        
        # OpenAI/Azure API handling
        if model == "gpt-4o-mini":
            print("Using Azure OpenAI API")
            self.HTTP_CLIENT = httpx.Client(proxy="http://127.0.0.1:8456")
            self.client = AzureOpenAI(
                api_key=api_key_dict.get('azure_key', ''),
                api_version="2024-07-01-preview",
                azure_endpoint=api_key_dict.get('azure_endpoint', ''),
                http_client=self.HTTP_CLIENT
            )
        # deepseek API handling
        elif model == "DeepSeek-R1":
            self.url = 'https://deepseek.ctyun.zgci.ac.cn:10443/v1'
            keys = api_key_dict.get(model, [])
            self.client = OpenAI(
                base_url=self.url,
                api_key=keys[key_idx],
                    )
        elif model in ["DeepSeek-R1-671B", "DeepSeek-R1-Distill-32B"]:
            self.url = 'https://madmodel.cs.tsinghua.edu.cn/v1/chat/completions'
            keys = api_key_dict.get(model, [])
            self.client = OpenAI(
                base_url=self.url,
                api_key=keys[key_idx],
                    )
        elif model in ["deepseek-chat", "deepseek-reasoner"]:
            self.url = 'https://api.deepseek.com'
            keys = api_key_dict.get(model, [])
            self.client = OpenAI(
                base_url=self.url,
                api_key=keys[key_idx],
                    )
        elif model in ["qwen-plus", "qwen-max"]:
            self.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            keys = api_key_dict.get(model, [])
            self.client = OpenAI(
                base_url=self.url,
                api_key=keys[key_idx],
                    )
        # SiliconFlow API handling
        else:
            self.url = "https://api.siliconflow.cn/v1/chat/completions"
            keys = api_key_dict.get("silicon_flow_keys", [])
            if not keys or key_idx >= len(keys):
                raise ValueError("Invalid key index or missing API keys")
            self.headers = {
                "Authorization": f"Bearer {keys[key_idx]}",
                "Content-Type": "application/json"
            }


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
                if payload["model"] == "gpt-4o-mini":
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
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
                
                elif payload["model"].startswith("DeepSeek") or payload["model"] == "deepseek-chat":
                    response = self.client.chat.completions.create(
                        model=payload["model"],
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

                elif payload["model"] == "deepseek-reasoner":
                    response = self.client.chat.completions.create(
                        model=payload["model"],
                        messages=payload["messages"],
                        max_tokens=payload["max_tokens"],
                        temperature=payload["temperature"],
                        top_p=payload["top_p"],
                        frequency_penalty=payload["frequency_penalty"],
                        n=payload["n"],
                        timeout=60
                    )
                    content = response.choices[0].message.content
                    reasoning_content = response.choices[0].message.reasoning_content
                    self.q_token += response.usage.prompt_tokens
                    self.a_token += response.usage.completion_tokens
                    if content:
                        if reasoning_content:
                            return {"choices": [{"message": {"content": content, "reasoning_content": reasoning_content}}]}
                        else:
                            return {"choices": [{"message": {"content": content}}]}
                        
                elif payload["model"] in ["qwen-plus", "qwen-max"]:
                    response = self.client.chat.completions.create(
                        model=payload["model"],
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
                            
                else:
                    response = requests.request("POST", self.url, json=payload, headers=self.headers)
                    response = response.json()
                    content = response["choices"][0]["message"]["content"]
                    self.q_token += response["usage"]["prompt_tokens"]
                    self.a_token += response["usage"]["completion_tokens"]
                    if content:
                        return {"choices": [{"message": {"content": content}}]}

            except requests.RequestException as e:
                print(f"API request failed (attempt {try_num + 1}/{max_try}): {e}")
            except KeyError as e:
                print(f"Unexpected API response structure: {e}")

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
    llm = LLM_api(model="Qwen/Qwen2.5-7B-Instruct", key_idx=0)
    # 单线程调用示例
    text = llm.get_text(content="你好")
    print("单线程结果:", text)

    # 多线程调用示例
    contents = ["你好", "こんにちは", "Hello", "안녕하세요", "Bonjour", "Hola", "Ciao", "Привет", "مرحبا", "你好"]
    responses = llm.get_multi_text(contents)
    print("多线程结果:")
    for i, response in enumerate(responses):
        print(f"内容 {contents[i]} 的响应: {response}")