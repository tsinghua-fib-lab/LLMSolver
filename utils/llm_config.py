# LLM API Configuration Dictionary
api_config_dict = {
    "gpt-4o-mini": {
        "base_url": "https://api.openai.com",
        "api_key": "API_KEY",
        "model": "gpt-4o-mini",
        "stream": False,
        "api_type": "azure",
        "api_version": "2024-07-01-preview",
        "azure_endpoint": "",
        "proxy": "http://127.0.0.1:8456"
    },
    "deepseek-chat": {
        "base_url": "https://api.deepseek.com",
        "api_key": "API_KEY",
        "model": "deepseek-chat",
        "stream": False,
        "api_type": "openai"
    },
    "deepseek-reasoner": {
        "base_url": "https://api.deepseek.com",
        "api_key": "API_KEY",
        "model": "deepseek-reasoner",
        "stream": False,
        "api_type": "openai"
    },
    "DeepSeek-R1": {
        "base_url": "https://deepseek.ctyun.zgci.ac.cn:10443/v1",
        "api_key": "API_KEY",
        "model": "DeepSeek-R1",
        "stream": False,
        "api_type": "openai"
    },
    "DeepSeek-R1-671B": {
        "base_url": "https://madmodel.cs.tsinghua.edu.cn/v1/chat/completions",
        "api_key": "API_KEY",
        "model": "DeepSeek-R1-671B",
        "stream": False,
        "api_type": "openai"
    },
    "DeepSeek-R1-Distill-32B": {
        "base_url": "https://madmodel.cs.tsinghua.edu.cn/v1/chat/completions",
        "api_key": "API_KEY",
        "model": "DeepSeek-R1-Distill-32B",
        "stream": False,
        "api_type": "openai"
    },
    "qwen-plus": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "API_KEY",
        "model": "qwen-plus",
        "stream": False,
        "api_type": "openai"
    },
    "qwen-max": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "API_KEY",
        "model": "qwen-max",
        "stream": False,
        "api_type": "openai"
    },
    "qwq-plus-latest": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "API_KEY",
        "model": "qwq-plus-latest",
        "stream": True,
        "api_type": "openai"
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "base_url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key": "API_KEY",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "stream": False,
        "api_type": "siliconflow"
    }
}

# Default model configuration
default_model_config = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "max_tokens": 4096,
    "temperature": 1.0,
    "top_p": 0.7,
    "frequency_penalty": 0,
    "n": 1,
    "stream": False
}

# Model type mapping
model_type_mapping = {
    "azure": ["gpt-4o-mini"],
    "openai": ["deepseek-chat", "deepseek-reasoner", "DeepSeek-R1", "DeepSeek-R1-671B", 
               "DeepSeek-R1-Distill-32B", "qwen-plus", "qwen-max", "qwq-plus-latest"],
    "siliconflow": ["Qwen/Qwen2.5-7B-Instruct"]
}

def get_model_config(model_name):
    """Get configuration for specified model"""
    return api_config_dict.get(model_name, {})

def get_api_key(model_name):
    """Get API key for specified model"""
    config = get_model_config(model_name)
    return config.get("api_key", "")

def get_model_type(model_name):
    """Get API type for specified model"""
    for api_type, models in model_type_mapping.items():
        if model_name in models:
            return api_type
    return "siliconflow"  # Default type
