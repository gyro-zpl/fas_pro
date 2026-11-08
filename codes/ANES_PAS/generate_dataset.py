import pandas as pd
import csv
import numpy as np
from utils import ol3, base_info, part_info
from biden_round2few import assess

def evaluate_range_accuracy(prediction, ground_truth):
    """
    判断预测值和真实值是否在同一个评分区间内
    
    Args:
        prediction (int/float): 预测值，范围[0, 999]
        ground_truth (int/float): 真实值，范围[0, 999]
        
    Returns:
        int: 如果两个值在同一区间返回1，否则返回0
        
    区间划分：
        [0, 20): 冷淡或不利
        [20, 40): 有点冷淡或不利  
        [40, 60): 中性
        [60, 80): 有点温暖或有利
        [80, 100): 相当温暖或有利
        [100, ...): 未知/异常值
    """
    def get_range_index(value):
        """获取数值所在的区间索引"""
        value = float(value)
        if 0 <= value < 20:
            return 0
        elif 20 <= value < 40:
            return 1
        elif 40 <= value < 60:
            return 2
        elif 60 <= value < 80:
            return 3
        elif 80 <= value < 100:
            return 4
        else:  # 包括 >= 100 和其他异常值如999
            return 5
    
    pred_range = get_range_index(prediction)
    gt_range = get_range_index(ground_truth)
    
    return 1 if pred_range == gt_range else 0

def target_mapping(pred_target):
    if pred_target == 'Mike Pence':
        return 'ftpence1'
    elif pred_target == 'Andrew Yang':
        return 'ftyang1'
    elif pred_target == 'Nancy Pelosi':
        return 'ftpelosi1'
    elif pred_target == 'Marco Rubio':
        return 'ftrubio1'
    elif pred_target == 'Alexandria Ocasio-Cortez':
        return 'ftocasioc1'
    elif pred_target == 'Nikki Haley':
        return 'fthaley1'
    elif pred_target == 'Clarence Thomas':
        return 'ftthomas1'
    elif pred_target == 'Dr. Anthony Fauci':
        return 'ftfauci1'
    elif pred_target == 'Donald Trump':
        return 'fttrump1'
    elif pred_target == 'Barack Obama':
        return 'ftobama1'
    elif pred_target == 'Joe Biden':
        return 'ftbiden1'
    else:
        return None
def assessment(row, pred_target):
    # 创建政治人物名称到字段名的映射
    politician_mapping = {
        'Mike Pence': 'ftpence1',
        'Andrew Yang': 'ftyang1', 
        'Nancy Pelosi': 'ftpelosi1',
        'Marco Rubio': 'ftrubio1',
        'Alexandria Ocasio-Cortez': 'ftocasioc1',
        'Nikki Haley': 'fthaley1',
        'Clarence Thomas': 'ftthomas1',
        'Dr. Anthony Fauci': 'ftfauci1',
        'Donald Trump': 'fttrump1',
        'Barack Obama': 'ftobama1',
        'Joe Biden': 'ftbiden1'
    }
    
    # 创建字段名到政治人物名称的反向映射
    reverse_mapping = {v: k for k, v in politician_mapping.items()}
    
    # 获取pred_target对应的字段名
    pred_field = target_mapping(pred_target)
    
    # 为所有政治人物创建评价字符串，但排除pred_target对应的人物
    politician_assessments = {}
    for field_name, politician_name in reverse_mapping.items():
        if field_name != pred_field:  # 排除pred_target对应的政治人物
            politician_assessments[politician_name] = assess(row[field_name]) + politician_name + ". "
    
    # 按照原有的变量名创建评价
    pence = politician_assessments.get('Mike Pence', '')
    yang = politician_assessments.get('Andrew Yang', '')
    pelosi = politician_assessments.get('Nancy Pelosi', '')
    rubio = politician_assessments.get('Marco Rubio', '')
    ocasioc = politician_assessments.get('Alexandria Ocasio-Cortez', '')
    haley = politician_assessments.get('Nikki Haley', '')
    thomas = politician_assessments.get('Clarence Thomas', '')
    fauci = politician_assessments.get('Dr. Anthony Fauci', '')
    trump = politician_assessments.get('Donald Trump', '')
    obama = politician_assessments.get('Barack Obama', '')
    biden = politician_assessments.get('Joe Biden', '')

    
    pred = pred_target
    pred_value = row[target_mapping(pred_target)]
    print(pred_value)
    
    # 动态构建assess_description，排除pred_target对应的政治人物
    # 获取所有有效的政治人物评价
    valid_assessments = []
    for politician_name, assessment_text in politician_assessments.items():
        if assessment_text:  # 如果不为空字符串
            valid_assessments.append(assessment_text.rstrip('. '))
    
    # 构建描述字符串
    if len(valid_assessments) >= 4:
        # 如果有4个或更多政治人物，使用原有的格式
        assess_description = f'''This person {valid_assessments[0]} and {valid_assessments[1]}, while at the same time {valid_assessments[2]} and {valid_assessments[3]}.
Besides, this person {' and '.join(valid_assessments[4:])}
''' if len(valid_assessments) > 4 else f'''This person {valid_assessments[0]} and {valid_assessments[1]}, while at the same time {valid_assessments[2]} and {valid_assessments[3]}.
'''
    elif len(valid_assessments) >= 2:
        # 如果有2-3个政治人物
        assess_description = f'''This person {valid_assessments[0]} and {valid_assessments[1]}'''
        if len(valid_assessments) > 2:
            assess_description += f''', while at the same time {' and '.join(valid_assessments[2:])}.
'''
        else:
            assess_description += '''.
'''
    elif len(valid_assessments) == 1:
        # 如果只有1个政治人物
        assess_description = f'''This person {valid_assessments[0]}.
'''
    else:
        # 如果没有有效的政治人物评价
        assess_description = "This person's political attitudes are being analyzed.\n"
    

    question = f'''According to the information, can you help me decide how would this person rate {pred_target}?  
Your response should consist of just one number between [0, 100] to reflect the person's attitude, without any additional text, explanation or even a space letter.
Here is an example of a required response that you should follow: 
    if you think this person has a fairly cold or unfavorable feeling, you should response JUST a integer number like 25. (or more/less favorable you think, the number may be bigger/smaller)
So your response should be like 25
'''
    

    return assess_description, question, pred_value
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
        pred_target = 'Donald Trump'
        for i, row in enumerate(reader):
            
            base_description = base_info(row)
            part_description = part_info(row, pred_target)
            assess_description, question, gt = assessment(row, pred_target)
            # print('================================')
            # print(base_description)
            # print('================================')
            # print(part_description)
            # print('================================')
            # print(assess_description)
            # print('================================')
            # print(question)
            # print('================================')
            # print(gt)
            # print('================================')
            # print(question)
            # print(gt)
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

if __name__ == "__main__":
    main()