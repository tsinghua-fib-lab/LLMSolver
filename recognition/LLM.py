import requests
import time
import httpx
import asyncio
from openai import OpenAI, AzureOpenAI
from API_key import api_key_dict

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
            keys = api_key_dict.get("DeepSeek-R1", [])
            self.client = OpenAI(
                base_url=self.url,
                api_key=keys[key_idx],
                    )
        elif model == "DeepSeek-R1-671B" or model == "DeepSeek-R1-Distill-32B":
            self.url = 'https://madmodel.cs.tsinghua.edu.cn/v1/chat/completions'
            keys = api_key_dict.get("DeepSeek-R1-671B", [])
            self.client = OpenAI(
                base_url=self.url,
                api_key=keys[key_idx],
                    )
        elif model == "deepseek-chat" or model == "deepseek-reasoner":
            self.url = 'https://api.deepseek.com'
            keys = api_key_dict.get("deepseek-chat", [])
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

    def get_response(self):
        """Handles API requests with retry mechanism"""
        max_try = 10
        sleep_time = 1
        max_sleep = 20

        for try_num in range(max_try):
            try:
                if self.payload["model"] == "gpt-4o-mini":
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=self.payload["messages"],
                        max_tokens=self.payload["max_tokens"],
                        temperature=self.payload["temperature"],
                        top_p=self.payload["top_p"],
                        frequency_penalty=self.payload["frequency_penalty"],
                        n=self.payload["n"],
                        timeout=60
                    )
                    content = response.choices[0].message.content
                    self.q_token += response.usage.prompt_tokens
                    self.a_token += response.usage.completion_tokens
                    if content:
                        return {"choices": [{"message": {"content": content}}]}
                
                elif self.payload["model"].startswith("DeepSeek") or self.payload["model"] == "deepseek-chat":
                    response = self.client.chat.completions.create(
                        model=self.payload["model"],
                        messages=self.payload["messages"],
                        max_tokens=self.payload["max_tokens"],
                        temperature=self.payload["temperature"],
                        top_p=self.payload["top_p"],
                        frequency_penalty=self.payload["frequency_penalty"],
                        n=self.payload["n"],
                        timeout=60
                    )
                    content = response.choices[0].message.content
                    self.q_token += response.usage.prompt_tokens
                    self.a_token += response.usage.completion_tokens
                    if content:
                        return {"choices": [{"message": {"content": content}}]}

                elif self.payload["model"] == "deepseek-reasoner":
                    response = self.client.chat.completions.create(
                        model=self.payload["model"],
                        messages=self.payload["messages"],
                        max_tokens=self.payload["max_tokens"],
                        temperature=self.payload["temperature"],
                        top_p=self.payload["top_p"],
                        frequency_penalty=self.payload["frequency_penalty"],
                        n=self.payload["n"],
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

            except requests.RequestException as e:
                print(response.json())
                print(f"API request failed (attempt {try_num + 1}/{max_try}): {e}")
            except KeyError as e:
                print(f"Unexpected API response structure: {e}")

            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 2, max_sleep)

        print("Warning: TOO MANY TRIES, EMPTY RESPONSE")
        return {"choices": [{"message": {"content": ""}}]}

    def get_text(self, content=""):
        """Gets text response from LLM API"""
        self.set_content(content)
        response = self.get_response()
        return response.get('choices', [{}])[0].get('message', {}).get('content', "")

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
    text = llm.get_text(content="你好")
    print(text)
