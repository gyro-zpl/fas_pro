import os
#COLUMNS = ["natenvir", "natheal", "natcity"]
COLUMNS = ["nateduc", "natdrug", "natcrime"]
API_URL = os.getenv("DIFY_API_URL", "http://10.113.178.150:2001/v1/chat-messages")
API_KEY_NONERAG = os.getenv("DIFY_API_KEY", "app-GhDluzpqgOPjYS8XXJDER0n9")
API_KEY_RAG = os.getenv("DIFY_API_KEY_RAG", "app-TFkxG9dj0YC5Fj0jUv11XctG")