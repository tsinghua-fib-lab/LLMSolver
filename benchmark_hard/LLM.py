import requests
import time
import httpx
import asyncio
from openai import OpenAI
from benchmark_hard.llm_config import api_config_dict


class LLM_api:
    def __init__(self,
                 model="deepseek",
                 max_tokens=4096,
                 stop=None,
                 temperature=1,
                 top_p=0.7,
                 frequency_penalty=0,
                 n=1,
                 timeout=60):

        self.payload = {
            "model": api_config_dict[model]['model'],
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

        self.client = OpenAI(api_key=api_config_dict[model].get('api_key', ''),
                             base_url=api_config_dict[model].get('base_url', ''),
                             timeout=timeout)

        self.q_token = 0
        self.a_token = 0

    def reset_token(self):
        self.q_token = 0
        self.a_token = 0

    def get_token(self):
        return self.q_token, self.a_token

    def get_response(self, content: str):
        """Handles API requests with retry mechanism"""
        max_try = 10
        sleep_time = 1
        max_sleep = 20

        messages = [{'role': 'user', 'content': content.strip()}, ]
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
                        messages=messages,
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
                        messages=messages,
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
                            return {
                                "choices": [{"message": {"content": content, "reasoning_content": reasoning_content}}]}
                        else:
                            return {"choices": [{"message": {"content": content}}]}

                else:
                    response = requests.request("POST", self.url, json=self.payload, headers=self.headers)
                    response = response.json()
                    content = response["choices"][0]["message"]["content"]
                    self.q_token += response["usage"]["prompt_tokens"]
                    self.a_token += response["usage"]["completion_tokens"]
                    if content:
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
        response = self.get_response(content)
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

    def print_usage(self):
        """Prints API token usage"""
        print(f"Prompt tokens: {self.q_token}")
        print(f"Completion tokens: {self.a_token}")


if __name__ == "__main__":
    llm = LLM_api(model="deepseek-chat", )
    text = llm.get_text(content="你好")
    print(text)
