import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Import thư viện Deep Learning (Keras)
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

def load_and_preprocess_data(dataset_path, target_size=(224, 224)):
    """
    Tiền xử lý dữ liệu: Đọc ảnh, resize và gán nhãn dựa vào file XML.
    Nhãn 1 = Có người, Nhãn 0 = Không có người.
    """
    print("1. Đang đọc và tiền xử lý ảnh...")
    train_img_dir = os.path.join(dataset_path, 'train', 'JPEGImages')
    train_ann_dir = os.path.join(dataset_path, 'train', 'Annotations')
    
    X_data = []
    y_labels = []
    
    img_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in img_files:
        # Bước 1.1: Đọc và thay đổi kích thước ảnh (Resize)
        img_path = os.path.join(train_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Chuyển hệ màu từ BGR (chuẩn của OpenCV) sang RGB (chuẩn của mạng CNN)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size) # VGG16 yêu cầu ảnh 224x224
        X_data.append(img_resized)
        
        # Bước 1.2: Phân tích file XML để lấy nhãn
        xml_name = os.path.splitext(img_name)[0] + '.xml'
        xml_path = os.path.join(train_ann_dir, xml_name)
        
        label = 0 # Mặc định ban đầu là 0 (Không có người)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name.lower() == 'person':
                    label = 1
                    break # Tìm thấy 1 người là đủ, thoát vòng lặp ngay
                    
        y_labels.append(label)
        
    # Chuyển danh sách sang ma trận NumPy để tính toán tốc độ cao
    X_data = np.array(X_data)
    y_labels = np.array(y_labels)
    
    # Chuẩn hóa giá trị pixel theo chuẩn của mô hình VGG16
    X_data = preprocess_input(X_data)
    
    print(f"   -> Hoàn tất! Kích thước ma trận ảnh: {X_data.shape}, Số nhãn: {len(y_labels)}")
    return X_data, y_labels

def extract_features_pretrained(X_data):
    """
    Đưa ảnh qua mạng VGG16 để "ép" thành các vector đặc trưng.
    """
    print("2. Đang tải kiến trúc mạng VGG16...")
    # include_top=False: Chặt bỏ lớp phân loại ở cuối, chỉ lấy phần trích xuất đặc trưng
    # pooling='avg': Ép ma trận đặc trưng thành một mảng 1 chiều (vector) gồm 512 con số
    model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    
    print("3. Đang chạy dữ liệu qua mạng CNN (có thể mất vài phút)...")
    # Tự động chia nhỏ (batch_size=32) để không bị tràn RAM
    features = model.predict(X_data, batch_size=32)
    
    print(f"   -> Hoàn tất trích xuất! Kích thước ma trận đặc trưng: {features.shape}")
    return features

def save_features_to_disk(features, labels, filename_prefix):
    """
    Lưu đặc trưng ra file .npy để nộp bài hoặc dùng cho lần sau.
    """
    # Đảm bảo thư mục tồn tại
    os.makedirs('features', exist_ok=True) 
    np.save(f"features/{filename_prefix}_X.npy", features)
    np.save(f"features/{filename_prefix}_y.npy", labels)
    print("4. Đã lưu thành công các file .npy vào thư mục 'features/'")