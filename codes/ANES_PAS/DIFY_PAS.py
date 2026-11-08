import dashscope
from http import HTTPStatus
import os
import requests
import json
import csv
import argparse
import re
from generate_dataset import base_info, part_info, assessment
# Read API keys from environment variables to avoid hardcoding
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

def dify_request_non_stream(prompt: str, conversation_id = None):
    """Non-stream request to Dify. Return the 'answer' string if present, otherwise the full JSON response."""
    api_url = os.environ.get("DIFY_API_URL", "")
    api_key = os.environ.get("DIFY_API_KEY", "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "inputs": {},
        "query": prompt,
        "response_mode": "blocking",
        "conversation_id": conversation_id,
        "user": "gyro"
    }

    r = requests.post(api_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    try:
        obj = r.json()
    except ValueError:
        # Not JSON
        return r.text

    #print(obj['conversation_id'])
    # Typical Dify response: {"answer": "..."} or more complex structures
    #print(obj)
    if isinstance(obj, dict) and "answer" in obj:
        return obj["answer"], obj['conversation_id']

    
    return obj, None

def clean_output(raw_output):
    """
    清洗DIFY输出，提取数字评分
    处理各种可能的输出格式，包括：
    1. 纯数字：如 "-65"
    2. 带解释的文本：如 "Based on... I would rate Donald Trump as -65."
    3. 包含大括号的JSON格式输出
    4. 拒绝回答的情况：返回0
    返回：(清洗后的输出, 是否成功提取数字)
    """
    if not raw_output or not isinstance(raw_output, str):
        return "0", False
    
    # 检查是否是拒绝回答的情况
    refusal_keywords = [
        "cannot provide",
        "cannot give",
        "unable to",
        "not able to",
        "refuse to",
        "decline to",
        "won't provide",
        "can't provide",
        "is there something else",
        "something else i can help"
    ]
    
    if any(keyword in raw_output.lower() for keyword in refusal_keywords):
        return "0", False
    
    # 首先尝试提取大括号内的内容（JSON格式）
    json_match = re.search(r'\{[^}]*\}', raw_output)
    if json_match:
        json_content = json_match.group()
        try:
            parsed = json.loads(json_content)
            if 'answer' in parsed:
                raw_output = parsed['answer']
            elif 'fail' in parsed and parsed['fail']:
                return "0", False
        except:
            pass
    
    # 查找数字模式，包括负数
    # 匹配模式：可选负号 + 数字 + 可选小数部分
    number_patterns = [
        r'rate.*?(-?\d+(?:\.\d+)?)',  # "rate ... as -65" 模式
        r'score.*?(-?\d+(?:\.\d+)?)',  # "score ... -65" 模式
        r'rating.*?(-?\d+(?:\.\d+)?)',  # "rating ... -65" 模式
        r'(-?\d+(?:\.\d+)?)',  # 基本数字模式（整数或小数）
    ]
    
    for pattern in number_patterns:
        matches = re.findall(pattern, raw_output, re.IGNORECASE)
        if matches:
            # 取第一个匹配的数字
            number_str = matches[0]
            try:
                # 验证是否为有效数字
                float(number_str)
                return number_str, True
            except ValueError:
                continue
    
    # 如果没有找到数字，返回0
    return "0", False

def run_once(prompt: str, conversation_id = None):
    raw, conversation_id = dify_request_non_stream(prompt,conversation_id)
    print(raw)
    cleaned_output, success = clean_output(raw)
    
    return cleaned_output, conversation_id, success

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=str, default="Donald Trump")
    parser.add_argument("--input_path", type=str, default=f'/home/cyyuan/ACL2025/Data/Anes2020/selected_anes2020.csv')
    parser.add_argument("--out_path", type=str, default=f'/home/cyyuan/FAS_PRO/codes/ANES_PAS/correct.txt')
    return parser.parse_args()

def judgement(output, gt):
    output = float(output)
    gt = float(gt)
    if output == gt or (output>=100 and gt>=100) or abs(gt-output)<10:
        return 1
    else:
        if output // 20 == gt // 20:
            return 1
        else:
            return 0
def main(mission, input_path, out_path):
    with open(input_path, mode='r', newline='') as infile, open(out_path, 'w') as outfile:
        reader = csv.DictReader(infile)
        cnt = 0
        #predict
        correct = 0
        failed_extractions = 0  # 记录无法提取数字的情况
        pred_target = mission
        conversation_id = None
        for i, row in enumerate(reader):
            base_description = base_info(row)
            part_description = part_info(row, pred_target)
            assess_description, question, gt = assessment(row, pred_target)
            sim_type = "\nSimulation type: PAS"
            prompt = sim_type+base_description + part_description + question

            output, conversation_id, success = run_once(prompt, conversation_id)
            print("prompt: ", prompt)

            exit()
            # print("output: ", output)
            # print("conversation_id: ", conversation_id)
            
            #print("output: ", output)
            if not success:
                failed_extractions += 1
            
            correct += judgement(output, gt)
            outfile.write(f"No:{i+1}, gt:{gt}, output:{output}, success:{success}, correct:{correct}\n")
            #print(f"No:{i+1}, gt:{gt}, output:{output}, correct:{correct}")
            # if i>1000:
            #     break
            #break

        acc = round(correct/(i+1)*100,2)
        failed_rate = round(failed_extractions/(i+1)*100,2)
        with open(out_path, 'a') as outfile:
            outfile.write(f"correct:{correct}\n")
            outfile.write(f"total:{i+1}\n")
            outfile.write(f"failed_extractions:{failed_extractions}\n")
            outfile.write(f"Accuracy: {acc}%\n")
            f.write(f"Failed extraction rate: {failed_rate}%\n")
        print(f"Accuracy: {acc}%")
        print(f"Failed extraction rate: {failed_rate}%")
        outfile.close()
if __name__ == "__main__":
    parser = parser_args()
    main(parser.mission, parser.input_path, parser.out_path)