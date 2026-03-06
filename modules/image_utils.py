# File: modules/image_utils.py
import os
import numpy as np
import cv2 # Thư viện xử lý ảnh OpenCV
# Import keras/tensorflow hoặc pytorch tùy bạn dùng framework nào

def load_and_preprocess_data(dataset_path):
    """
    Hàm này nhận đường dẫn từ Kaggle, chui vào các thư mục con 
    để đọc ảnh, resize về cùng kích thước và chuẩn hóa (normalization).
    """
    print(f"Đang xử lý dữ liệu từ: {dataset_path}")
    X = []
    y = []
    
    # TODO: Viết logic dùng thư viện os để duyệt qua thư mục pos (có người) và neg (không có người)
    # Ví dụ: cv2.imread(ảnh), cv2.resize(ảnh, (128, 128)), chuẩn hóa / 255.0
    
    return np.array(X), np.array(y)

def extract_features_pretrained(X_data):
    """
    Theo yêu cầu đề bài, bạn dùng các mạng có sẵn (VGG16, ResNet...) để rút trích đặc trưng.
    """
    print("Đang chạy mô hình CNN để trích xuất đặc trưng...")
    # TODO: Khởi tạo model CNN pretrained, predict() để lấy feature vector
    features = [] 
    return np.array(features)

def save_features_to_disk(features, labels, filename_prefix):
    """
    Lưu đặc trưng ra file .npy để nộp bài theo đúng yêu cầu.
    """
    np.save(f"features/{filename_prefix}_X.npy", features)
    np.save(f"features/{filename_prefix}_y.npy", labels)
    print(f"Đã lưu file đặc trưng vào thư mục features/")