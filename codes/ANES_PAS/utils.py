import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
from biden_round2few import pp, load_mapping, base_info
      
def ol3(messages):
    stream = ollama.chat(
        model='llama3:8b',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

def ol31(messages):
    stream = ollama.chat(
        model='llama3.1:8b',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans
def part_info(row, pred_str='moneyorg', pred_str_description='given money to an organization concerned with a political or social issue'):
    #enrollment_features = ['meeting','moneyorg','protest','online', 'persuade','button']
    mapping = load_mapping('/home/cyyuan/ACL2025/Anes2020/mappings/mapping.json')

    meeting = pp(row['meeting']) + "attended a meeting to talk about political or social concerns. "
    moneyorg = pp(row['moneyorg']) + "given money to an organization concerned with a political or social issue. "
    protest = pp(row['protest']) + "joined in a protest march, rally, or demonstration. "
    online = pp(row['online']) + "posted a message or comment online about a political issue or campaign. "
    persuade = pp(row['persuade']) + "tried to persuade anyone to vote one way or another. "
    button = pp(row['button'])+ "worn a campaign button put a sticker on his/her car or placed a sign in window or in front of his/her house. "
    part_description = f'''
This person's participations on political activities are as follows: 
    {button}
    {online}
    {meeting}
    {protest}
    {persuade}
    {moneyorg}
'''

    return part_description

def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/Anes2020/'
    input_path = '/home/cyyuan/ACL2025/Data/Anes2020/selected_anes2020.csv'
    output_path = folder_path + 'A20par_descriptions.txt'
    df = pd.read_csv(input_path) 


    responses = []
    gts = []


    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        #predict
        correct = 0
        for i, row in enumerate(reader):
            
            base_description = base_info(row)
            # part_description = part_info(row)
            part_description, question, gt = part_info(row, pred_str, pred_str_description)

            prompt = base_description + part_description + question
            conversation_history = [{"role": "user", "content": prompt}]
            try:
                response = float(ol3(conversation_history))  # 尝试将返回值转换为 float
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue 
            gt = float(gt)

            responses.append(response)
            gts.append(gt)
            
            # print(f"No:{i+1}"+part_description)
            # print(f"The answer is:{gt}")
            if response == gt:
                correct = correct+1
            
    responses = []
    gts = []
    acc = correct/(i+1)
    acc = format(acc*100,".2f")
    print("total num: ", i, "\nacc: ", acc)
    responses = np.array(responses)  
    gts = np.array(gts)  

    with open('/home/cyyuan/ACL2025/Data/Anes2020/duolunC/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"moneyorg_round3_zero_ollama3:  {acc}%\n")

    print("write succesfully")



    with open(input_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        cnt = 0
        #predict
        correct = 0
        for i, row in enumerate(reader):
            
            base_description = base_info(row)
            # part_description = part_info(row)
            part_description, question, gt = part_info(row, pred_str, pred_str_description  )

            prompt = base_description + part_description + question
            conversation_history = [{"role": "user", "content": prompt}]
            try:
                response = float(ol31(conversation_history))  # 尝试将返回值转换为 float
            except (ValueError, TypeError) as e:
                # 捕获转换失败的情况并跳过当前循环
                print(f"Conversion failed for conversation: {conversation_history}, error: {e}")
                continue 
            gt = float(gt)

            responses.append(response)
            gts.append(gt)
            
            #print(part_description)
            #print(f"The answer is:{gt}")
            if response == gt:
                correct = correct+1
            print(f"No:{i}")

            
    acc = correct/(i+1)
    acc = format(acc*100,".2f")
    print("total num: ", i, "\nacc: ", acc)
    responses = np.array(responses)  
    gts = np.array(gts)  

    with open('/home/cyyuan/ACL2025/Data/Anes2020/duolunC/acc.txt', 'a') as file:  # 使用 'a' 模式可以追加内容
        file.write(f"moneyorg_round3_zero_ollama31:  {acc}%\n")

    print("write succesfully")
if __name__ == "__main__":
    main()

#     Here are some examples and answers:
# 1. This person, who ages 69, homeowner and has an income between $10,000 - $14,999. Besides, this person has a voting tendency: Probably would not vote. 
# This person has a summarized participation score 0.
# This person has a neutral feeling (score 60.0, in range of [40,60]) for rating to Mike Pence.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Andrew Yang. , while at the same time has a neutral feeling (score 60.0, in range of [40,60]) for rating to Clarence Thomas.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Dr. Anthony Fauci. .
# Besides, this person has a neutral feeling (score 60.0, in range of [40,60]) for rating to Joe Biden.  and has a neutral feeling (score 60.0, in range of [40,60]) for rating to Donald Trump. 
# The answer is 60

# 2. This person, who ages 22, homeowner and has an income between $75,000 - $79,999. Besides, this person has a voting tendency: Select someone to vote for. 
# This person has a summarized participation score 5.
# This person has a bit warm or favorable feeling (score 65.0, in range of [60,80]) for rating to Mike Pence.  and has a bit warm or favorable feeling (score 70.0, in range of [60,80]) for rating to Andrew Yang. , while at the same time has a bit cold or unfavorable feeling (score 40.0, in range of [20,40]) for rating to Clarence Thomas.  and has a bit warm or favorable feeling (score 80.0, in range of [60,80]) for rating to Dr. Anthony Fauci. .
# Besides, this person has a neutral feeling (score 50.0, in range of [40,60]) for rating to Joe Biden.  and has a cold or unfavorable feeling (score 0.0, in range of [0,20]) for rating to Donald Trump. 
# The answer is 40