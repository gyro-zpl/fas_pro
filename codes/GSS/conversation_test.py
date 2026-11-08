import dashscope
from http import HTTPStatus
import os
import requests
import json
import csv
from DIFY_none_RAG import run_once,dify_request_non_stream

def main():
    prompt = "I am gyro.Hello."
    raw, conversation_id = dify_request_non_stream(prompt,conversation_id=None)
    print(raw)

    prompt = "who am i?"
    raw, conversation_id = dify_request_non_stream(prompt,conversation_id=conversation_id)
    print(raw)

    prompt = "what is my name?"
    raw, conversation_id = dify_request_non_stream(prompt,conversation_id=conversation_id)
    print(raw)

    prompt = "what is my name?"
    raw, conversation_id = dify_request_non_stream(prompt,conversation_id=conversation_id)
    print(raw)
if __name__ == "__main__":
    main()