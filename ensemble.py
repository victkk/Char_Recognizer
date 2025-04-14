import pandas as pd
from collections import Counter
import os

# 在这里指定10个CSV文件的路径
csv_files = [
    'result/2025-04-14_16-51-52_freeze_0_resnet101/result.csv',
    'result/2025-04-14_17-17-58_freeze_1_resnet101/result.csv',
    'result/2025-04-14_17-50-51_freeze_2_resnet101/result.csv',
    'result/2025-04-14_18-11-33_freeze_3_resnet101/result.csv',
    'result/2025-04-14_18-31-45_freeze_4_resnet101/result.csv',
    'result/2025-04-14_18-57-36_freeze_5_resnet101/result.csv',
    'result/2025-04-14_19-18-04_freeze_6_resnet101/result.csv',
]

import pandas as pd
from collections import Counter
import os
import math

def ensemble_voting(csv_files, output_file='ensemble_results.csv', 
                    top_inconsistent=10, inconsistency_metric='entropy',
                    save_inconsistency_scores=False):
    """
    对多个模型的预测结果进行投票集成
    
    参数:
        csv_files (list): 包含模型预测结果的CSV文件路径列表
        output_file (str): 保存集成结果的文件路径
        top_inconsistent (int): 展示标签预测不一致性最高的图片数量
        inconsistency_metric (str): 用于排序的不一致性指标 ('entropy', 'consistency_ratio', 'unique_labels')
        save_inconsistency_scores (bool): 是否保存所有图片的不一致性得分到CSV文件
    """
    # 验证指标选择
    valid_metrics = ['entropy', 'consistency_ratio', 'unique_labels']
    if inconsistency_metric not in valid_metrics:
        print(f"警告: 无效的不一致性指标 '{inconsistency_metric}'，使用默认值 'entropy'")
        inconsistency_metric = 'entropy'
    
    # 存储所有图片的投票结果
    voting_results = {}
    
    # 记录有效的模型数量
    valid_model_count = 0
    processed_files = []

    # 读取所有CSV文件并进行投票
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"警告: 文件 {csv_file} 不存在，跳过。")
            continue
            
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            valid_model_count += 1
            processed_files.append(os.path.basename(csv_file))
            
            for index, row in df.iterrows():
                file_name = row['file_name']
                file_code = row['file_code']
                
                if file_name not in voting_results:
                    voting_results[file_name] = []
                
                voting_results[file_name].append(file_code)
        
        except Exception as e:
            print(f"读取 {csv_file} 时出错: {e}")

    if valid_model_count == 0:
        print("错误: 没有有效的模型预测文件。")
        return

    # 对每张图片找出最多票的标签，并计算一致性指标
    final_results = []
    inconsistency_scores = []
    
    # 统计信息
    total_consistency_ratio = 0
    total_unique_labels = 0
    total_entropy = 0
    
    for file_name, codes in voting_results.items():
        vote_counts = Counter(codes)
        most_common = vote_counts.most_common()
        
        # 选择出现次数最多的标签
        most_common_code = most_common[0][0]
        most_common_count = most_common[0][1]
        
        # 计算一致性指标
        # 方法1：计算最高票数占总票数的比例（越高表示越一致）
        consistency_ratio = most_common_count / len(codes)
        total_consistency_ratio += consistency_ratio
        
        # 方法2：计算不同标签的数量（越少表示越一致）
        unique_labels = len(vote_counts)
        total_unique_labels += unique_labels
        
        # 方法3：计算熵（越低表示越一致）
        entropy = 0
        for label, count in vote_counts.items():
            prob = count / len(codes)
            entropy -= prob * math.log2(prob)
        total_entropy += entropy
        
        # 添加结果
        final_results.append({
            'file_name': file_name, 
            'file_code': most_common_code
        })
        
        # 记录不一致性得分及其详细信息
        inconsistency_scores.append({
            'file_name': file_name,
            'consistency_ratio': consistency_ratio,
            'unique_labels': unique_labels,
            'entropy': entropy,
            'predictions': dict(vote_counts),
            'most_common_code': most_common_code
        })

    # 创建最终的DataFrame并保存为CSV
    final_df = pd.DataFrame(final_results)
    final_df.sort_values(by='file_name', inplace=True)  # 按文件名排序
    final_df.to_csv(output_file, index=False)

    # 计算平均值
    avg_consistency_ratio = total_consistency_ratio / len(voting_results)
    avg_unique_labels = total_unique_labels / len(voting_results)
    avg_entropy = total_entropy / len(voting_results)

    # 输出基本统计信息
    print("\n===== 集成投票结果统计 =====")
    print(f"已处理 {len(final_results)} 张图片，来自 {valid_model_count} 个有效模型预测。")
    print(f"已处理的模型文件: {', '.join(processed_files)}")
    print(f"结果已保存到: '{output_file}'")
    print(f"平均一致性比例: {avg_consistency_ratio:.4f} (越高越一致)")
    print(f"平均不同标签数: {avg_unique_labels:.2f} (越低越一致)")
    print(f"平均熵值: {avg_entropy:.4f} (越低越一致)")
    
    # 根据选择的指标进行排序
    if inconsistency_metric == 'consistency_ratio':
        # 一致性比例越低越不一致
        sort_key = lambda x: x['consistency_ratio']
        reverse_sort = False
        metric_desc = "一致性比例 (越低越不一致)"
    elif inconsistency_metric == 'unique_labels':
        # 不同标签数量越多越不一致
        sort_key = lambda x: x['unique_labels']
        reverse_sort = True
        metric_desc = "不同标签数量 (越多越不一致)"
    else:  # entropy
        # 熵越高越不一致
        sort_key = lambda x: x['entropy']
        reverse_sort = True
        metric_desc = "熵值 (越高越不一致)"
    
    # 按照选择的不一致性指标排序
    inconsistency_scores.sort(key=sort_key, reverse=reverse_sort)
    
    # 显示标签预测不一致的图片
    print(f"\n===== 标签预测最不一致的图片 (基于{metric_desc}) =====")
    cnt = 0
    for item in inconsistency_scores:
        if(item["unique_labels"]==7):
            cnt+=1
    print(cnt)
        # 显示最不一致的top_inconsistent张图片
    for i, item in enumerate(inconsistency_scores[:top_inconsistent]):
        print(f"\n{i+1}. 文件名: {item['file_name']}")
        print(f"   最终选择的标签: {item['most_common_code']}")
        print(f"   一致性比例: {item['consistency_ratio']:.2f} (最高票数/总票数)")
        print(f"   不同标签数量: {item['unique_labels']}")
        print(f"   熵值: {item['entropy']:.4f}")
        
        # 格式化输出每个标签的投票数
        predictions = item['predictions']
        prediction_str = ", ".join([f"标签{label}: {count}票" for label, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True)])
        print(f"   预测分布: {prediction_str}")
    
    # 保存不一致性得分
    if save_inconsistency_scores:
        scores_df = pd.DataFrame([
            {
                'file_name': item['file_name'],
                'file_code': item['most_common_code'],
                'consistency_ratio': item['consistency_ratio'],
                'unique_labels': item['unique_labels'],
                'entropy': item['entropy']
            }
            for item in inconsistency_scores
        ])
        
        # 保存不一致性得分到CSV
        inconsistency_file = os.path.splitext(output_file)[0] + '_inconsistency_scores.csv'
        scores_df.to_csv(inconsistency_file, index=False)
        print(f"\n所有图片的不一致性得分已保存到: '{inconsistency_file}'")



# 执行投票
ensemble_voting(
    csv_files, 
    output_file='final_ensemble_results.csv', 
    top_inconsistent=10,
    inconsistency_metric='entropy',  # 可选: 'entropy', 'consistency_ratio', 'unique_labels'
    save_inconsistency_scores=True
)
