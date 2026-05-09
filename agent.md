BẢN ĐẶC TẢ KỸ THUẬT CHO AI AGENT: HIỆN THỰC PIPELINE PHÁT HIỆN NGƯỜI ĐI BỘ (INRIAPERSON)
Mục tiêu: Xây dựng một pipeline Machine Learning hoàn chỉnh để phát hiện người đi bộ trong ảnh.
Môi trường: Python, chạy trên Jupyter Notebook (Google Colab hoặc Local).
Yêu cầu đầu ra: Nhận đầu vào là 1 bức ảnh bất kỳ có chứa người -> Model tự động nhận diện và dùng OpenCV để vẽ khung bao (bounding box) hình chữ nhật quanh đối tượng người có trong ảnh.
Nguồn: Kaggle dataset jcoral02/inriaperson.
Cấu trúc lưu trữ:
Thư mục JPEGImages/: Chứa toàn bộ hình ảnh gốc (Positive và Negative).
Thư mục Annotations/: Chứa các file siêu dữ liệu định dạng .xml tương ứng với ảnh có chứa người.
Định dạng Annotation: Dữ liệu theo chuẩn PASCAL VOC. Các thông tin quan trọng nằm trong thẻ <object>. Tên nhãn nằm ở thẻ <name> (là "person"), và tọa độ khung bao tuyệt đối nằm ở các thẻ con của <bndbox> gồm: <xmin>, <ymin>, <xmax>, <ymax>.
Yêu cầu Agent lập trình tuần tự theo các khối (block) logic sau:
Bước 1: Setup & Data Loading
Khởi tạo môi trường, import các thư viện cần thiết: numpy, cv2 (OpenCV), matplotlib, xml.etree.ElementTree, sklearn, skimage.
Tích hợp Kaggle API để tải dataset trực tiếp: lệnh !kaggle datasets download -d jcoral02/inriaperson và giải nén tệp.
Bước 2: XML Parsing & ROI Extraction
Viết hàm duyệt qua thư mục Annotations/.
Sử dụng xml.etree.ElementTree để parse các file XML.
Trích xuất tọa độ [xmin, ymin, xmax, ymax] của các thẻ <name>person</name>.
Ánh xạ sang tên file trong JPEGImages/ để đọc ma trận ảnh.
Cắt (crop) vùng ảnh chứa người (Positive samples) dựa trên tọa độ trên và resize chuẩn hóa về 64x128 pixel.
Trích xuất các vùng ảnh âm bản (Negative samples) từ các khu vực không chứa người (IoU = 0) và cũng resize về 64x128 pixel.
Bước 3: Feature Extraction (Trích xuất đặc trưng)
Xây dựng module hỗ trợ 2 phương pháp trích xuất đặc trưng, cho phép người dùng tùy chọn (cấu hình qua biến):
Cấu hình 1 (HOG): Sử dụng skimage.feature.hog hoặc cv2.HOGDescriptor(). Cấu hình cell 8x8, block 16x16.
Cấu hình 2 (Pre-trained CNN): Sử dụng mạng CNN (VD: VGG16, ResNet50 bỏ top layer). Cho ảnh ROI đi qua mạng và flatten tensor đầu ra thành vector 1D.
I/O Requirement: Lưu toàn bộ tập ma trận đặc trưng huấn luyện này vào ổ đĩa dưới định dạng features.npy hoặc features.h5 để tái sử dụng.
Bước 4: Classifier Training (Huấn luyện phân loại)
Load data từ tệp .npy hoặc .h5 lên bộ nhớ RAM.
Dùng train_test_split để chia tập dữ liệu.
Cài đặt mô hình LinearSVC từ sklearn.svm.
Gọi hàm .fit() với tập đặc trưng và nhãn (1 cho person, 0 cho background).
(Optional) Tích hợp bước Hard Negative Mining: dùng mô hình vừa train quét trên ảnh không có người, nếu mô hình báo có người (False Positive) thì add ngay ROI đó vào tập Negative và train lại để tăng độ chính xác.
Bước 5: Object Detection Inference (Chạy suy luận)
Xây dựng hàm xử lý ảnh thực tế: detect_pedestrian(image_path, model)
Đọc ảnh đầu vào.
Image Pyramid: Tạo vòng lặp thu nhỏ ảnh dần dần theo một tỷ lệ scale (vd: thu nhỏ 1.05 lần mỗi bước).
Sliding Window: Ở mỗi tầng của tháp ảnh, trượt một cửa sổ ảo kích thước 64x128 qua toàn bộ ảnh với stride cho trước (vd: 4 hoặc 8 pixels).
Tại mỗi điểm dừng, crop nội dung cửa sổ -> Trích xuất đặc trưng (HOG/CNN) -> Đưa vào model LinearSVC để lấy confidence score.
Nếu nhận diện là 1 (Person), chiếu tọa độ cửa sổ ảo về tỷ lệ ảnh gốc (nhân với tỷ lệ scale hiện tại) và lưu tọa độ `` cùng điểm confidence.
Bước 6: Post-processing (NMS) & Visualization
Dữ liệu từ Bước 5 sẽ sinh ra rất nhiều hộp giới hạn chồng chéo lên nhau trên cùng 1 người.
Áp dụng thuật toán Non-Maximum Suppression (NMS) (Có thể dùng hàm cv2.dnn.NMSBoxes  hoặc viết NMS bằng tay với IoU threshold = 0.3 - 0.5) để chỉ giữ lại 1 bounding box tối ưu nhất cho mỗi người.
Dùng lệnh cv2.rectangle vẽ khung bao màu xanh lá (0, 255, 0) dựa trên tọa độ đã qua lọc NMS.
Dùng matplotlib.pyplot.imshow hiển thị kết quả cuối cùng.
