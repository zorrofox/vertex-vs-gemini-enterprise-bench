import os

# 测试不依赖真实 .env：在导入 discovery_api.config 之前固定环境变量
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_PROJECT_NUMBER", "123")
os.environ.setdefault("AGENT_SEARCH_ENGINE", "test-engine")
os.environ.setdefault("AGENT_SEARCH_ASSISTANT", "default_assistant")
os.environ.setdefault("DISCOVERY_LOCATION", "global")
