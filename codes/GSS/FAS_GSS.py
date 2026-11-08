import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
import outlines
from outlines.inputs import Chat
from openai import OpenAI
from jinja2 import Template

from typing import List, Literal
from pydantic import BaseModel, validator

def format_output_with_template(data):
    """使用Jinja2模板格式化输出"""
    template_str = """{
  {%- for row in rows %}
    {%- if not (row is iterable and row|length == 3) %}
    /* ERROR: each row must be an iterable of exactly 3 items */
    {%- else %}
    [
      {%- for item in row %}
        {{ item|tojson }}{{ "," if not loop.last else "" }}
      {%- endfor %}
    ]{{ "," if not loop.last else "" }}
    {%- endif %}
  {%- endfor %}
}"""
    template = Template(template_str)
    return template.render(rows=data)

class BatchResponse(BaseModel):
    rows: List[List[Literal["1", "2", "3"]]]

    @validator("rows")
    def validate_rows(cls, rows):
        for row in rows:
            if len(row) != 3:
                raise ValueError("each row must contain exactly 3 items")
        return rows

def clean_model_output(text):
    """清理大模型输出，提取JSON数据"""
    print(f"DEBUG: 原始模型输出: {repr(text)}")
    
    # 移除可能的markdown代码块标记
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    
    # 移除前后的空白字符
    text = text.strip()
    print(f"DEBUG: 清理后输出: {repr(text)}")
    
    # 尝试直接解析JSON
    try:
        result = json.loads(text)
        print(f"DEBUG: 直接JSON解析成功: {result}")
        return result
    except json.JSONDecodeError as e:
        print(f"DEBUG: 直接JSON解析失败: {e}")
    
    # 如果直接解析失败，尝试提取数组部分
    import re
    
    # 查找形如 [["1","2","3"], ["2","1","3"]] 的模式
    array_pattern = r'\[\s*\[.*?\]\s*\]'
    matches = re.findall(array_pattern, text, re.DOTALL)
    print(f"DEBUG: 数组模式匹配: {matches}")
    
    if matches:
        try:
            # 尝试解析找到的第一个匹配项
            result = json.loads(matches[0])
            print(f"DEBUG: 数组模式解析成功: {result}")
            return result
        except json.JSONDecodeError as e:
            print(f"DEBUG: 数组模式解析失败: {e}")
    
    # 如果还是失败，尝试手动构建数组
    # 查找所有形如 ["1","2","3"] 的子数组
    row_pattern = r'\[\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*\]'
    rows = re.findall(row_pattern, text)
    print(f"DEBUG: 行模式匹配: {rows}")
    
    if rows:
        # 构建标准化的数组
        result = []
        for row in rows:
            result.append([str(row[0]), str(row[1]), str(row[2])])
        print(f"DEBUG: 手动构建成功: {result}")
        return result
    
    # 尝试更宽松的模式匹配
    # 查找任何包含三个数字的模式
    loose_pattern = r'(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)'
    loose_matches = re.findall(loose_pattern, text)
    print(f"DEBUG: 宽松模式匹配: {loose_matches}")
    
    if loose_matches:
        result = []
        for match in loose_matches:
            result.append([str(match[0]), str(match[1]), str(match[2])])
        print(f"DEBUG: 宽松模式构建成功: {result}")
        return result
    
    raise ValueError("无法从模型输出中提取有效的JSON数据")

def parse_and_validate_output(text, batch_size):
    try:
        # 首先尝试清理和解析输出
        data = clean_model_output(text)
    except Exception as e:
        raise ValueError(f"model output is not valid JSON: {str(e)}") from e
    
    try:
        validated = BatchResponse(rows=data)
    except Exception as e:
        raise ValueError("model output does not match required schema") from e
    
    if len(validated.rows) != batch_size:
        raise ValueError(f"row count {len(validated.rows)} does not match batch_size {batch_size}")
    
    return validated.rows

def ol_outlines(messages):
    model = outlines.from_ollama(ollama.client, "llama3:8b")
    prompt = Chat([
        {"role": "user", "content": messages}
    ])
    response = model("what is 1+1?")
    return response
def ol(messages):

    stream = ollama.chat(
        model='llama3:8b',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

def generate_background(batchsize):

    database_background = '''
    The General Social Survey 2024 (GSS2024), conducted in 2020 by NORC at the University of Chicago, collects data on American society to track trends in opinions, attitudes, and behaviors. 
    Funded by the NSF, it covers topics like civil liberties, crime, and social mobility, enabling researchers to analyze societal changes over decades and compare the U.S. to other nations.
    '''

# The distribution ratios of the options selected by respondents on some other questions are as follows:
# The resources the country spends on the military, armaments and defense:
# [1: 0.285, 2: 0.389, 3: 0.325]
# The resources the country spends on halting the rising crime rate:
# [1: 0.726, 2: 0.196, 3: 0.078]
# The resources the country spends on improving the nation's education system:
# [1: 0.75, 2: 0.186, 3: 0.064]
    role_play_info = '''
   You are a statistician and a social survey expert. Your task is to generate a survey dataset: '''
    task = f'''
The sample batch size is {batchsize}. The dataset should include 3 demographic variables for a respondent, whose attitude towards the following questions: 

The resources the country spends on improving and protecting the environment is 
    1.too little. 
    2.the right amount. 
    3.too much. 
The resources the country spends on improving and protecting the nation's health is
    1.too little. 
    2.the right amount. 
    3.too much. 
The resources the country spends on solving the problems of big cities
    1.too little. 
    2.the right amount. 
    3.too much. 
 
    
    Here are some examples of data record list: 
['1', '1', '2']
['1', '2', '1']
['1', '1', '1']
['1', '1', '2']
['1', '1', '1']

    Explanations: ['1', '2', '3'] (which represent the choices of the three questions are "too little","the right amount" and "too much")

    Choose your answers only from the options provided. After generating, only show the data you generated without additional words.
    The format must be a JSON string representing a three-dimensional array. Also, make sure that it is an array of arrays with no objects, like in a spreadsheet.
    Remember, the records should closely reflect the GSS dataset."
'''
    return role_play_info+database_background+task
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/GSS/'
    input_path = folder_path + 'selected_GSS.csv'
    output_path = folder_path + 'testForpic.txt'
    df = pd.read_csv(input_path) 
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        prompt = generate_background(50)
        print(prompt)
        # 初始化对话历史，包含首轮用户提示
        conversation_history = [{"role": "user", "content": prompt}]
        for i, row in enumerate(reader):
    
            # outfile.write(prompt+'\n')
            # 使用带历史的对话消息
            messages = conversation_history
            print(f"No:{i} generating...")
            aa = ol( messages)
            outfile.write(aa+'\n')
            # 将本轮 assistant 输出加入历史，供下一轮使用
            conversation_history.append({"role": "assistant", "content": aa})
            #aa = get_sample_inform(row)
            
            #batchsize 20 times 20
            if i >= 20:
                print("Done! ")
                break
def FAS(batch_size, epoch_num):
    prompt = generate_background(batch_size)
    conversation_history = [{"role": "user", "content": prompt}]
    aa = None
    for i in range(1, epoch_num + 1):
        messages = conversation_history
        aa = ol(messages)
        print(f"DEBUG: 第{i}轮模型输出: {repr(aa)}")
        conversation_history.append({"role": "assistant", "content": aa})
    if aa is None:
        raise RuntimeError("no model output produced")
    
    # 解析并验证输出
    validated_data = parse_and_validate_output(aa, batch_size)
    
    # 使用Jinja2模板格式化输出
    formatted_output = format_output_with_template(validated_data)
    
    return formatted_output

if __name__ == "__main__":
    print(FAS(20, 20))