# 安装依赖: pip install paddlepaddle paddleocr opencv-python numpy

from paddleocr import PaddleOCR
import cv2
import numpy as np
import os

def detect_text_with_dbnet(image_path, output_path=None):
    """
    使用DBNet模型从图像中检测文本区域，将检测到的最佳边界框扩大20%，并保存边界框内的图像
    
    Args:
        image_path: 图像文件路径
        output_path: 输出结果图像路径，如果为None则不保存
        
    Returns:
        最可能是文本的边界框 [x, y, width, height]（原始尺寸，未扩大）
    """
    # 检测文本
    result = ocr.ocr(image_path, cls=False)
    image = cv2.imread(image_path)
    # 如果没有检测到文本

    if result[0] == None or len(result) == 0 or len(result[0]) == 0:
        print("未检测到任何文本")
        cv2.imwrite(output_path,image)
        return None
    
    # 读取原始图像
    
    img_height, img_width = image.shape[:2]
    
    # 提取所有检测到的文本框和对应的置信度
    bboxes = []
    confidence_scores = []
    
    for line in result[0]:
        bbox_points = line[0]  # 四个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        confidence = line[1]   # 检测置信度
        
        # 转换多边形为矩形边界框
        x_min = min(point[0] for point in bbox_points)
        y_min = min(point[1] for point in bbox_points)
        x_max = max(point[0] for point in bbox_points)
        y_max = max(point[1] for point in bbox_points)
        
        width = x_max - x_min
        height = y_max - y_min
        
        bboxes.append([int(x_min), int(y_min), int(width), int(height)])
        confidence_scores.append(confidence)
    
    max_value = 0
    for i, value in enumerate(confidence_scores):
        if value[1] > max_value:
            max_value = value[1]
            best_bbox_idx = i
    best_bbox = bboxes[best_bbox_idx]
    best_score = confidence_scores[best_bbox_idx]
    
    print(f"检测到最可能的文本框，置信度: {best_score[1]:.4f}")
    
    # 将边界框扩大20%
    x, y, w, h = best_bbox
    
    # 计算扩大后的尺寸
    new_w = int(w * 1.5)
    new_h = int(h * 1.5)
    
    # 计算新的左上角坐标，使边界框居中扩展
    new_x = max(0, x - (new_w - w) // 2)
    new_y = max(0, y - (new_h - h) // 2)
    
    # 确保边界框不超出图像边界
    new_x = min(new_x, img_width - new_w)
    new_y = min(new_y, img_height - new_h)
    new_w = min(new_w, img_width - new_x)
    new_h = min(new_h, img_height - new_y)
    
    expanded_bbox = [new_x, new_y, new_w, new_h]
    
    # 如果需要保存结果
    if output_path:
        # 裁剪扩大后的边界框区域
        cropped_image = image[new_y:new_y+new_h, new_x:new_x+new_w]
        
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存裁剪后的图像
        cv2.imwrite(output_path, cropped_image)
        print(f"已将裁剪后的图像保存至 {output_path}")
    
    # 返回原始的边界框（未扩大）
    return best_bbox

# 示例使用
if __name__ == "__main__":
    ocr = PaddleOCR(use_angle_cls=False, lang='ch', rec=False, 
                    det_model_dir=None, use_gpu=False)
    image_path = "dataset/mchar_test_a/005969.png"  # 替换为你的图像路径
    for img_path in os.listdir("dataset/mchar_test_a"):
        img_path = os.path.join("dataset/mchar_test_a", img_path)
        output_path = os.path.join("dataset/mchar_test_a_cut", os.path.basename(img_path))
        bbox = detect_text_with_dbnet(img_path, output_path)
        
        # if bbox:
        #     print(f"检测到的最佳文本框: [x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}]")
