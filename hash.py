import imagehash
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
# 计算图像哈希值
def compute_hashes(directory):
    results = []
    for filepath in tqdm(Path(directory).glob('*.*')):
        if filepath.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            try:
                img_hash = imagehash.phash(Image.open(filepath))
                results.append({'file': str(filepath), 'hash': str(img_hash)})
            except Exception as e:
                print(f"处理{filepath}时出错: {e}")
    return pd.DataFrame(results)

# 计算两个文件夹的图像哈希
df1 = compute_hashes('/data/zhangzicheng/workspace/study/Char_Recognizer/dataset/mchar_train')
df2 = compute_hashes('/data/zhangzicheng/workspace/study/Char_Recognizer/dataset/mchar_test_a')

# 找出哈希值相同的图片
duplicates = pd.merge(df1, df2, on='hash', suffixes=('_1', '_2'))
print(duplicates)