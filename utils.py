import yaml
import os
import datetime
import re



def sanitize_folder_name(name):
    """清理说明字符串中的非法字符"""
    # 替换空格为下划线
    name = name.strip().replace(' ', '_')
    # 移除非字母、数字、下划线和连字符的字符
    name = re.sub(r'[^\w\-_]', '', name)
    # 移除连续的下划线
    name = re.sub(r'_+', '_', name)
    return name

def create_timestamped_folder(description, base_dir="result",has_timestamp=True):
    """
    根据当前时间和说明字符串创建文件夹
    :param description: 说明文字（会被清理特殊字符）
    :param base_dir: 基础目录（默认为当前目录）
    :return: 创建的文件夹完整路径
    """
    try:
        # 获取当前时间并格式化
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 清理说明文字
        clean_desc = sanitize_folder_name(description)
        if not clean_desc:
            raise ValueError("Description cannot be empty after sanitization")
        
        # 组合文件夹名称
        if(has_timestamp):
            folder_name = f"{timestamp}_{clean_desc}"
        else:
            folder_name = f"{clean_desc}"
        # 构建完整路径
        full_path = os.path.join(base_dir, folder_name)
        
        # 创建目录（包括父目录）
        os.makedirs(full_path, exist_ok=False)
        print(f"文件夹创建成功：{full_path}")
        return full_path
    
    except FileExistsError:
        print(f"错误：文件夹已存在 - {full_path}")
    except PermissionError:
        print(f"错误：没有权限在目录 {base_dir} 中创建文件夹")
    except ValueError as ve:
        print(f"错误：{ve}")
    except Exception as e:
        print(f"未知错误：{str(e)}")
    return None