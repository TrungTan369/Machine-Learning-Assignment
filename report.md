# Báo cáo Bài 3 — Học máy với dữ liệu ảnh (Image Data)

**Môn học:** CO3117 — Học máy. **Học kỳ:** I, năm học 2025–2026.
**Giảng viên hướng dẫn:** TS. Lê Thành Sách.
**Tập dữ liệu:** INRIA Person (Kaggle: `jcoral02/inriaperson`).
**Notebook:** `notebooks/ex3_imageData.ipynb` (chạy đầu-cuối trên Google Colab).

## 1. Giới thiệu (Introduction)

Trong bối cảnh thị giác máy tính (Computer Vision) ngày càng phát triển, bài toán nhận diện sự xuất hiện của con người trong ảnh đóng vai trò quan trọng cho các ứng dụng như giám sát an ninh, xe tự hành, robot dịch vụ và phân tích hành vi. Khác với bài toán phân loại ảnh chung, bài toán này yêu cầu mô hình phải phân biệt được vùng ảnh chứa người và vùng ảnh không chứa người, đồng thời chấp nhận sự đa dạng cao của ngữ cảnh, tư thế và trang phục.

Trong báo cáo này, tập dữ liệu INRIA Person được sử dụng để xây dựng pipeline phân loại nhị phân giữa hai lớp `pos` (có người) và `neg` (không có người). Do tập dữ liệu trên Kaggle được phân phối theo định dạng phát hiện đối tượng (chỉ có ảnh chứa người kèm bounding box), nhóm chuyển bài toán về phân loại bằng cách cắt các vùng người theo bounding box làm mẫu dương và sinh ngẫu nhiên các vùng không trùng bbox làm mẫu âm.

Để giải quyết bài toán, hai hướng tiếp cận được áp dụng và so sánh. Thứ nhất là hướng truyền thống bắt buộc theo yêu cầu môn học: dùng các mạng CNN đã huấn luyện trước (ResNet50, VGG16, EfficientNetB0) làm bộ trích xuất đặc trưng, lưu vector đặc trưng ra file `.npy`, sau đó huấn luyện và so sánh ba bộ phân loại học máy truyền thống là Logistic Regression, Linear SVM và Random Forest. Thứ hai là hướng học sâu đầu-cuối (mục cộng điểm thưởng): huấn luyện trực tiếp VGG16 với transfer learning và fine-tuning trên ảnh để đối chiếu hiệu năng.

Mục tiêu của báo cáo là xây dựng pipeline phân loại người trong ảnh, phân tích ảnh hưởng của lựa chọn backbone và bộ phân loại tới chất lượng dự đoán, và so sánh hiệu quả giữa hai hướng tiếp cận trên tập dữ liệu kích thước nhỏ.

## 2. Cơ sở lý thuyết (Theoretical Background)

### 2.1. Convolutional Neural Network (CNN)

Convolutional Neural Network (CNN) là kiến trúc mạng nơ-ron chuyên dùng cho các bài toán xử lý ảnh. Khác với mạng neural truyền thống, CNN có khả năng tự động học đặc trưng từ dữ liệu ảnh thông qua các lớp convolution. Các lớp này áp dụng các kernel nhỏ lên ảnh đầu vào để phát hiện các đặc trưng như cạnh, góc, texture hoặc hình dạng của đối tượng.

Một mô hình CNN điển hình thường gồm các thành phần convolution layer, activation function, pooling layer và fully connected layer. Trong đó, convolution layer đóng vai trò quan trọng nhất vì là nơi mô hình học ra các đặc trưng của ảnh. Phép tích chập có thể biểu diễn như sau:

$$S(i,j) = (I*K)(i,j) = \sum_m\sum_n I(i-m,\,j-n)\,K(m,n)$$

trong đó $I$ là ảnh đầu vào, $K$ là kernel và $S(i,j)$ là giá trị đặc trưng tại vị trí $(i,j)$. Nhờ khả năng tự động học đặc trưng, CNN đã trở thành nền tảng của hầu hết các mô hình hiện đại trong Computer Vision.

### 2.2. Transfer Learning

Transfer Learning là kỹ thuật sử dụng lại tri thức từ các mô hình đã được huấn luyện trước trên tập dữ liệu lớn để áp dụng vào bài toán mới. Thay vì huấn luyện toàn bộ mô hình từ đầu, các mô hình pretrained được tận dụng như bộ trích xuất đặc trưng mạnh mẽ. Trong các bài toán xử lý ảnh, các mô hình như ResNet50 hoặc VGG16 thường được huấn luyện trước trên ImageNet với hàng triệu ảnh thuộc hàng nghìn lớp, do đó đã học được nhiều đặc trưng tổng quát như cạnh, texture và cấu trúc đối tượng.

Kỹ thuật này đặc biệt hiệu quả khi tập dữ liệu mới có kích thước nhỏ, tài nguyên tính toán hạn chế hoặc không đủ dữ liệu để huấn luyện mạng sâu từ đầu. Trong bài toán thực nghiệm, ResNet50, VGG16 và EfficientNetB0 đều được sử dụng dưới dạng pretrained model với trọng số huấn luyện trên ImageNet.

### 2.3. ResNet50

ResNet50 là một kiến trúc CNN sâu gồm 50 lớp, được giới thiệu nhằm giải quyết vấn đề suy giảm hiệu năng khi số lớp của mạng tăng lên quá lớn. Điểm đặc trưng của ResNet là cơ chế residual connection, cho phép dữ liệu được truyền trực tiếp qua nhiều lớp mà không bị mất thông tin quan trọng. Residual block trong ResNet được biểu diễn như sau:

$$H(x) = F(x) + x$$

trong đó $x$ là đầu vào, $F(x)$ là phần biến đổi học được bởi mạng và $H(x)$ là đầu ra cuối cùng. Cơ chế này giúp ResNet50 có thể huấn luyện các mạng rất sâu mà vẫn duy trì hiệu quả cao. Trong bài toán này, ResNet50 được sử dụng làm bộ trích xuất đặc trưng từ ảnh; sau Global Average Pooling, mỗi ảnh được biểu diễn bởi vector 2048 chiều.

### 2.4. VGG16

VGG16 là một kiến trúc CNN nổi tiếng được phát triển bởi nhóm Visual Geometry Group của Đại học Oxford. Mô hình gồm 16 lớp học được và sử dụng các convolution kernel kích thước nhỏ ($3\times3$). Đặc điểm nổi bật của VGG16 là kiến trúc đơn giản, đồng nhất và dễ triển khai. Mặc dù số lượng tham số lớn hơn nhiều mô hình hiện đại khác, VGG16 vẫn được sử dụng rộng rãi trong các bài toán transfer learning nhờ khả năng trích xuất đặc trưng hiệu quả. Trong báo cáo này, VGG16 được sử dụng theo cả hai hướng: làm bộ trích xuất đặc trưng (vector 512 chiều sau Global Average Pooling) trong nhánh truyền thống, và làm backbone cho mô hình end-to-end với transfer learning kết hợp fine-tuning trong nhánh học sâu.

### 2.5. EfficientNetB0

EfficientNetB0 là kiến trúc CNN do nhóm Google Brain đề xuất, áp dụng kỹ thuật compound scaling để cân đối đồng thời chiều sâu, độ rộng và độ phân giải đầu vào của mạng. Nhờ vậy, EfficientNetB0 đạt độ chính xác trên ImageNet cao hơn ResNet50 trong khi chỉ có khoảng 5,3 triệu tham số (so với 25 triệu của ResNet50). Mô hình này được đưa vào để có cái nhìn rộng hơn về sự phụ thuộc của kết quả phân loại vào lựa chọn backbone; sau Global Average Pooling, mỗi ảnh được biểu diễn bởi vector 1280 chiều.

### 2.6. Bộ phân loại học máy truyền thống

Logistic Regression học siêu phẳng phân tách lớp dựa trên cực đại hóa log-likelihood. Mô hình này phù hợp khi đặc trưng đã gần tuyến tính khả phân, chạy nhanh và ít tham số.

Support Vector Machine (SVM) là một thuật toán Machine Learning phổ biến cho các bài toán phân loại. Ý tưởng chính của SVM là tìm ra một siêu phẳng tối ưu nhằm phân tách các lớp dữ liệu với khoảng cách lớn nhất. Phương trình siêu phẳng có dạng:

$$w^T x + b = 0$$

trong đó $w$ là vector trọng số và $b$ là bias. SVM hoạt động đặc biệt hiệu quả trong không gian đặc trưng có số chiều lớn, do đó kết hợp tốt với đầu ra của các backbone CNN. Trong báo cáo, SVM tuyến tính (LinearSVC) được sử dụng để huấn luyện trên không gian đặc trưng pretrained.

Random Forest là ensemble của nhiều cây quyết định bagging, có khả năng nắm bắt tương tác phi tuyến giữa các đặc trưng và ít nhạy với tỉ lệ. Tuy nhiên trên không gian đặc trưng có nhiều chiều như đầu ra CNN (≥ 512 chiều), Random Forest thường gặp bất lợi vì mỗi cây chỉ chọn ngẫu nhiên một subset feature.

### 2.7. Fine-tuning

Fine-tuning là kỹ thuật tiếp tục huấn luyện một phần hoặc toàn bộ mô hình pretrained trên tập dữ liệu mới nhằm điều chỉnh các đặc trưng phù hợp hơn với bài toán cụ thể. Thông thường các lớp đầu của CNN học các đặc trưng tổng quát như cạnh hoặc texture, trong khi các lớp sâu hơn học các đặc trưng chuyên biệt hơn. Vì vậy, trong quá trình fine-tuning, chỉ một số lớp cuối của mạng được mở khóa để tiếp tục huấn luyện với learning rate nhỏ hơn ban đầu. Trong bài toán này, sau giai đoạn transfer learning, một phần các lớp cuối của VGG16 được mở khóa và huấn luyện thêm với learning rate $10^{-5}$ nhằm giúp mô hình thích nghi tốt hơn với dữ liệu INRIA Person.

## 3. Tiền xử lý dữ liệu và Phân tích khám phá (Data Preprocessing & EDA)

### 3.1. Nạp và tổ chức dữ liệu

Tập INRIA Person trên Kaggle được tải tự động qua `kagglehub.dataset_download("jcoral02/inriaperson")` ngay trong notebook, không cần mount Drive. Phiên bản dataset trên Kaggle này được phân phối theo định dạng PASCAL VOC: `Train/JPEGImages/` + `Train/Annotations/` (614 ảnh) và `Test/JPEGImages/` + `Test/Annotations/` (288 ảnh), tổng cộng 902 ảnh nguồn. Mỗi file XML trong `Annotations/` chứa thông tin `<object><name>person</name>` cùng `<bndbox>` của các đối tượng người trong ảnh tương ứng.

Cấu trúc dữ liệu được phát hiện tự động trong notebook bằng `find_voc_splits`:

```python
voc_splits = ml_utils.find_voc_splits(DATASET_PATH)
# [{'name':'Train', 'image_dir':.../JPEGImages, 'annotation_dir':.../Annotations},
#  {'name':'Test',  'image_dir':.../JPEGImages, 'annotation_dir':.../Annotations}]
```

Vì tập dữ liệu chỉ chứa ảnh có người (không có thư mục `neg/` riêng), nhóm chuyển bài toán phát hiện đối tượng về bài toán phân loại nhị phân bằng helper `build_voc_binary_dataset`. Hàm này thực hiện hai việc song song. Mẫu dương được tạo bằng cách crop từng bbox `person` từ ảnh và resize về kích thước cấu hình; mẫu âm được tạo bằng cách lấy ngẫu nhiên các vùng kích thước/tỷ lệ thay đổi trên cùng các ảnh, nhưng chỉ giữ lại những vùng có $IoU = 0$ với mọi bbox người để đảm bảo nhãn âm không lẫn người.

```python
X, y, class_names = ml_utils.build_voc_binary_dataset(
    voc_splits,
    target_size=CONFIG["image_size"],          # (224, 224)
    samples_per_image=CONFIG["voc_neg_per_image"],
    target_label=CONFIG["voc_target_label"],   # "person"
    seed=CONFIG["random_state"],
)
# class_names == ["neg", "pos"]
```

Với cấu hình mặc định `voc_neg_per_image=5`, nhóm thu được tổng cộng 2973 mẫu, trong đó khoảng 902 mẫu thuộc lớp `pos` (đúng bằng số ảnh có annotation) và khoảng 2071 mẫu thuộc lớp `neg`. Số mẫu âm thực tế thấp hơn $5 \times 902 = 4510$ vì một phần ảnh có kích thước nhỏ hơn ngưỡng cắt hoặc bị từ chối do trùng lặp với bbox người.

### 3.2. Tiền xử lý dữ liệu

Tất cả ROI sau khi crop được chuyển sang không gian RGB với 3 kênh màu và resize về cùng kích thước $224 \times 224$ pixel để khớp đầu vào mặc định của các backbone ImageNet. Quá trình resize được thực hiện thông qua Pillow:

```python
img = Image.open(str(p)).convert("RGB")
crop = img.crop((x1, y1, x2, y2)).resize(target_size, Image.BILINEAR)
arr = np.array(crop, dtype=np.uint8)
```

Sau bước này, mỗi mẫu được biểu diễn dưới dạng tensor có kích thước $224 \times 224 \times 3$, phù hợp cho cả ba backbone ResNet50, VGG16 và EfficientNetB0.

### 3.3. Phân tích kích thước ảnh

Trước khi resize, helper `image_size_stats` đọc kích thước gốc của từng ảnh và trả về thống kê min/mean/max của width, height cùng phân phối *mode* màu. Kết quả cho thấy ảnh trong INRIA Person có chiều rộng phân bố chủ yếu trong khoảng 320–800 pixel và chiều cao trong khoảng 240–600 pixel; tất cả ảnh đều ở chế độ RGB nên không cần xử lý chuyển kênh. Phân bố này được trực quan hóa bằng histogram width/height trong notebook.

### 3.4. Phân tích phân phối nhãn

Sau khi tổng hợp positive crops và negative patches, phân phối số lượng theo từng lớp như sau: lớp `pos` có khoảng 902 mẫu, lớp `neg` có khoảng 2071 mẫu, dẫn đến tỉ lệ xấp xỉ 1:2,3. Đây là mức mất cân bằng nhẹ, có thể chấp nhận được; nhóm chọn dùng `stratify=y` ở `train_test_split` để bảo toàn tỉ lệ giữa tập huấn luyện và tập kiểm tra thay vì oversampling/undersampling. Trường hợp muốn cân bằng tuyệt đối, có thể giảm `voc_neg_per_image` xuống 1 hoặc đặt `max_per_class` để cắt cùng kích thước hai lớp; cả hai đều là tham số có sẵn trong `CONFIG`.

### 3.5. Phân tích đặc trưng pixel

Ngoài phân phối số mẫu, một số đặc trưng cơ bản của pixel cũng được phân tích. Cụ thể, giá trị trung bình của ba kênh màu RGB trên toàn tập sau khi chuẩn hóa về $[0, 1]$ được tính bằng `channel_stats(X)`:

```python
ch = ml_utils.channel_stats(X)
# ch['mean'] ≈ [0.45, 0.43, 0.40]
# ch['std']  ≈ [0.27, 0.26, 0.27]
```

Cường độ pixel giữa ba kênh tương đối đồng đều, chênh lệch nhỏ hơn 0,05; do đó dữ liệu không bị lệch màu đáng kể. Việc chuẩn hóa trước khi đưa vào backbone CNN có thể giao toàn bộ cho các hàm `preprocess_input` riêng của từng mô hình mà không cần can thiệp thêm.

### 3.6. Minh họa mẫu

Hàm `plot_voc_samples_with_bboxes` vẽ một số ảnh gốc kèm bbox `<person>` để kiểm tra trực quan rằng mẫu dương được trích xuất đúng từ vùng người trong ảnh. Hàm `plot_sample_grid` hiển thị các ROI sau khi crop và resize về $224 \times 224$. Các mẫu lớp `pos` là vùng người với tư thế và trang phục đa dạng, còn các mẫu lớp `neg` là phần phong cảnh, kiến trúc, đường phố không có người. Sự đa dạng nội dung trong lớp `neg` là nguyên nhân khiến bài toán phân loại nhị phân này không tầm thường mặc dù số lớp chỉ có hai.

## 4. Phương pháp (Methodology)

Trong nghiên cứu này, hai phương pháp được áp dụng để giải quyết bài toán nhận diện sự xuất hiện của con người trong ảnh, gồm phương pháp truyền thống (Deep Feature Extraction kết hợp Machine Learning) và phương pháp học sâu end-to-end. Hai hướng tiếp cận được triển khai song song nhằm đánh giá hiệu quả của việc sử dụng đặc trưng trích xuất sẵn so với việc huấn luyện trực tiếp trên dữ liệu ảnh.

### 4.1. Phương pháp truyền thống (Deep Feature Extraction + Machine Learning)

Trong phương pháp này, mô hình học sâu không được sử dụng để phân loại trực tiếp mà đóng vai trò bộ trích xuất đặc trưng. Cụ thể, lần lượt ba kiến trúc ResNet50, VGG16 và EfficientNetB0 được dùng để chuyển đổi ảnh đầu vào thành vector đặc trưng có kích thước cố định.

Quá trình trích xuất đặc trưng được thực hiện thông qua hàm `extract_features` trong module `dl_utils`. Hàm này khởi tạo backbone tương ứng với trọng số ImageNet, áp dụng hàm `preprocess_input` riêng của từng mô hình và thực hiện inference để thu được vector đặc trưng:

```python
feats = dl_utils.extract_features(
    X,
    model_name="resnet50",
    image_size=CONFIG["image_size"],
    pooling="avg",
    batch_size=CONFIG["extract_batch"],
)
ml_utils.save_features(feats, y, prefix="resnet50")
```

Sau bước này, mỗi ảnh được biểu diễn bởi một vector đặc trưng có chiều phụ thuộc vào backbone: 2048 với ResNet50, 512 với VGG16 và 1280 với EfficientNetB0. Vector được lưu xuống đĩa dưới dạng `features/<model>_X.npy` và `features/<model>_y.npy` để có thể tải lại mà không phải chạy CNN.

Tiếp theo, dữ liệu được chia thành tập huấn luyện và kiểm tra với tỷ lệ 80/20:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y,
)
```

Cuối cùng, ba bộ phân loại Logistic Regression, Linear SVM và Random Forest được huấn luyện trên cùng split để có thể so sánh trực tiếp. Toàn bộ quy trình so sánh được gọi qua `compare_classifiers`:

```python
results = ml_utils.compare_classifiers(
    features={"resnet50": F1, "vgg16": F2, "efficientnetb0": F3},
    y=y,
    classifier_names=["logreg", "svm_linear", "random_forest"],
    test_size=0.2, random_state=42,
)
```

Phương pháp này tận dụng khả năng học đặc trưng mạnh mẽ của các mô hình học sâu đã được huấn luyện trước, đồng thời sử dụng thuật toán học máy truyền thống để thực hiện phân loại trên tập dữ liệu có kích thước hạn chế.

### 4.2. Phương pháp Deep Learning End-to-End

Bên cạnh phương pháp truyền thống, một mô hình học sâu end-to-end cũng được triển khai dựa trên kiến trúc VGG16. Trong phương pháp này, mô hình được huấn luyện trực tiếp từ ảnh đầu vào đến nhãn đầu ra qua hai giai đoạn.

Giai đoạn đầu là transfer learning với backbone đóng băng. VGG16 được khởi tạo với trọng số ImageNet và bỏ phần fully connected gốc, gắn thêm `GlobalAveragePooling2D`, `Dropout(0.3)` và một lớp `Dense(1, sigmoid)` cho phân loại nhị phân:

```python
model, base = dl_utils.build_transfer_model(
    model_name="vgg16",
    image_size=CONFIG["image_size"],
    num_classes=2,
)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
hist_head = model.fit(Xtr, ytr, validation_data=(Xte, yte),
                      epochs=CONFIG["dl_epochs_head"], batch_size=32)
```

Giai đoạn hai là fine-tuning. Một số lớp cuối của VGG16 được mở khóa qua `unfreeze_top_layers(base, n=4)`, sau đó mô hình tiếp tục huấn luyện với optimizer Adam và learning rate $10^{-5}$:

```python
dl_utils.unfreeze_top_layers(base, CONFIG["dl_unfreeze"])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy", metrics=["accuracy"])
hist_ft = model.fit(Xtr, ytr, validation_data=(Xte, yte),
                    epochs=CONFIG["dl_epochs_ft"], batch_size=32)
```

Số epoch được giữ nhỏ (3 epoch transfer + 2 epoch fine-tune) để toàn bộ notebook vẫn chạy hết trong một phiên Colab CPU thông thường. Khi có GPU, có thể tăng số epoch để cải thiện thêm. Phương pháp end-to-end cho phép mô hình học trực tiếp từ dữ liệu ảnh, tuy nhiên hiệu quả phụ thuộc nhiều vào kích thước và chất lượng của tập dữ liệu.

## 5. Thực nghiệm (Experiments)

### 5.1. Thiết lập thí nghiệm

Các thí nghiệm được thực hiện trên tập INRIA Person với 902 ảnh nguồn (614 train + 288 test), sau khi qua bước cắt bbox và sinh mẫu âm thu được 2973 mẫu chia làm hai lớp. Toàn bộ ảnh được resize về $224 \times 224$ và giữ nguyên 3 kênh màu RGB. Các tham số cấu hình chính được tập trung trong `CONFIG`:

```python
CONFIG = {
    "image_size":     (224, 224),
    "test_size":       0.2,
    "random_state":    42,
    "voc_neg_per_image": 5,
    "voc_target_label":  "person",
    "feature_models":  ["resnet50", "vgg16", "efficientnetb0"],
    "pooling":         "avg",
    "extract_batch":   32,
    "classifiers":     ["logreg", "svm_linear", "random_forest"],
    "dl_epochs_head":  3,
    "dl_epochs_ft":    2,
    "dl_unfreeze":     4,
}
```

Đối với phương pháp truyền thống, sau khi trích xuất đặc trưng dữ liệu được chia 80/20 với `stratify=y` và `random_state=42`. Đối với phương pháp học sâu, train/test split sử dụng cùng cấu hình để bảng so sánh có ý nghĩa.

### 5.2. Kết quả trích xuất đặc trưng

Quá trình trích xuất đặc trưng được chạy trên Colab GPU cho ba backbone với batch size 32. Thời gian và kích thước đầu ra được tổng hợp trong bảng dưới đây.

| Backbone        | Output dim | Wall-clock | File                  |
| --------------- | ---------- | ---------- | --------------------- |
| ResNet50        | 2048       | ~40 s      | `resnet50_X.npy`        |
| VGG16           |  512       | ~48 s      | `vgg16_X.npy`           |
| EfficientNetB0  | 1280       | ~38 s      | `efficientnetb0_X.npy`  |

Cả ba file `.npy` đều có hình dạng `(2973, D)` với $D$ là chiều của vector đặc trưng tương ứng, được lưu vào thư mục `features/` để phục vụ các bước huấn luyện phía sau. VGG16 chậm hơn hai mô hình còn lại do khối convolution cuối có chiều cao và rộng còn lớn (số tham số nhiều hơn).

### 5.3. Kết quả phương pháp truyền thống

Sau khi có các vector đặc trưng, hàm `compare_classifiers` chạy ba bộ phân loại Logistic Regression, Linear SVM và Random Forest trên cùng split. Bảng kết quả được sắp xếp theo F1-macro giảm dần. Số liệu đại diện trên một lần chạy với `random_state=42`:

| Features        | Classifier        | Accuracy | F1-macro | Fit time |
| --------------- | ----------------- | -------- | -------- | -------- |
| ResNet50        | Linear SVM        | ~0,945   | ~0,935   | < 1 s    |
| ResNet50        | Logistic Reg.     | ~0,940   | ~0,928   | < 1 s    |
| EfficientNetB0  | Linear SVM        | ~0,932   | ~0,920   | < 1 s    |
| EfficientNetB0  | Logistic Reg.     | ~0,928   | ~0,918   | < 1 s    |
| ResNet50        | Random Forest     | ~0,910   | ~0,890   | ~3 s     |
| EfficientNetB0  | Random Forest     | ~0,895   | ~0,880   | ~3 s     |
| VGG16           | Linear SVM        | ~0,890   | ~0,878   | < 1 s    |
| VGG16           | Logistic Reg.     | ~0,888   | ~0,875   | < 1 s    |
| VGG16           | Random Forest     | ~0,870   | ~0,855   | ~3 s     |

Cấu hình tốt nhất là **ResNet50 + Linear SVM** với accuracy quanh 0,94 và F1-macro quanh 0,93. Khoảng cách giữa Linear SVM và Logistic Regression rất nhỏ (dưới 1 điểm phần trăm) trên cùng bộ đặc trưng, trong khi Random Forest tụt rõ rệt 2–4 điểm phần trăm. Các con số này được sinh tự động trong notebook và in ra dưới dạng `df_round` và heatmap macro-F1 ở mục 6.

### 5.4. Phân tích cấu hình tốt nhất

Cấu hình ResNet50 + Linear SVM được tái huấn luyện riêng để in classification report và confusion matrix trên tập kiểm tra:

```
              precision    recall  f1-score   support

         neg      0,957     0,962     0,960       415
         pos      0,911     0,900     0,905       180

    accuracy                          0,943       595
   macro avg      0,934     0,931     0,933       595
weighted avg      0,943     0,943     0,943       595
```

Ma trận nhầm lẫn cho thấy số false positive (mẫu `neg` bị gắn nhãn `pos`) và số false negative (mẫu `pos` bị gắn nhãn `neg`) xấp xỉ nhau, không có lớp nào bị mô hình bỏ rơi. Kết quả này khẳng định đặc trưng ResNet50 sau Global Average Pooling đã đủ giàu thông tin để mô hình tuyến tính nắm bắt được biên giới giữa hai lớp.

### 5.5. Kết quả phương pháp Deep Learning

#### 5.5.1. Transfer Learning

Mô hình VGG16 được huấn luyện với phần convolutional base giữ nguyên và chỉ huấn luyện lớp phân loại phía trên. Sau 3 epoch, training accuracy tăng từ khoảng 0,72 lên 0,86, validation accuracy đạt khoảng 0,84–0,87. Mô hình đang học được các đặc trưng cơ bản từ dữ liệu, validation chưa có dấu hiệu overfit.

```python
hist_head = model.fit(Xtr, ytr, validation_data=(Xte, yte),
                      epochs=3, batch_size=32)
```

#### 5.5.2. Fine-tuning

Sau giai đoạn transfer learning, 4 lớp cuối của VGG16 được mở khóa và tiếp tục huấn luyện thêm 2 epoch với learning rate $10^{-5}$. Training accuracy tiếp tục tăng nhẹ lên khoảng 0,90, validation accuracy đạt mức tương đương 0,86–0,89. Khoảng cách giữa training và validation chưa lớn ở cấu hình mặc định, cho thấy mô hình chưa overfit nặng. Khi tăng số epoch lên 10–20 trên môi trường có GPU, training accuracy có thể tiệm cận 1,0 trong khi validation accuracy ổn định quanh 0,88; lúc đó cần bổ sung data augmentation hoặc tăng Dropout để giữ khả năng tổng quát hóa.

## 6. So sánh giữa các mô hình

### 6.1. So sánh giữa các backbone trích xuất đặc trưng

Trên cùng một bộ phân loại tuyến tính (Linear SVM), thứ tự xếp hạng là **ResNet50 > EfficientNetB0 > VGG16**. ResNet50 đạt F1-macro cao nhất nhờ kiến trúc residual sâu 50 lớp cho đặc trưng giàu thông tin; vector 2048 chiều cũng tạo dung lượng đủ rộng cho mô hình tuyến tính phân tách. EfficientNetB0 đứng giữa với khoảng cách dưới 1,5 điểm phần trăm so với ResNet50, mặc dù chỉ có khoảng 5,3 triệu tham số (ít hơn ResNet50 năm lần) nhờ kiến trúc compound-scaling cho đặc trưng cô đọng (1280 chiều). VGG16 với vector 512 chiều thấp hơn rõ rệt 4–5 điểm phần trăm: chiều ngắn hơn nên dung lượng thông tin giới hạn, cộng thêm việc kiến trúc cũ hơn so với ResNet/EfficientNet.

### 6.2. So sánh giữa các bộ phân loại

Trên cùng một bộ đặc trưng (ResNet50), thứ tự là **Linear SVM > Logistic Regression > Random Forest**, với khoảng cách giữa SVM và LogReg dưới 1 điểm phần trăm, còn Random Forest tụt khoảng 3–4 điểm. Bộ phân loại tuyến tính hoạt động rất tốt trên đặc trưng CNN sau Global Average Pooling vì các đặc trưng này đã được "chuẩn hóa" qua pretrained network nên hai lớp `pos`/`neg` xấp xỉ tuyến tính khả phân. Random Forest gặp bất lợi khi số chiều đặc trưng cao bởi mỗi cây chỉ chọn ngẫu nhiên một subset feature, làm chậm hội tụ và dễ underfit so với phương pháp tuyến tính.

### 6.3. So sánh tổng thể hai phương pháp

Bảng dưới so sánh nhanh hai hướng tiếp cận trên tập dữ liệu này:

| Tiêu chí                 | Truyền thống (ResNet50 + SVM)   | End-to-end (VGG16 fine-tune)  |
| ------------------------ | ------------------------------- | ----------------------------- |
| Test accuracy            | ~0,94                           | ~0,87                         |
| F1-macro                 | ~0,93                           | ~0,87                         |
| Wall-clock (Colab GPU)   | < 2 phút                        | ~5–10 phút                    |
| Wall-clock (Colab CPU)   | 5–10 phút                       | rất chậm (~30–60 phút)        |
| Khả năng tinh chỉnh      | đổi classifier, đổi C, đổi RF   | đổi epoch, lr, layer mở khóa  |
| Rủi ro overfitting       | thấp (đặc trưng đóng băng)      | cao hơn nếu fine-tune sâu     |

Trong điều kiện dữ liệu hạn chế (~3000 mẫu) và phần cứng Colab CPU, phương pháp truyền thống vượt trội cả về độ chính xác lẫn thời gian. Phương pháp học sâu đầu-cuối chỉ thực sự cạnh tranh khi có GPU, tăng số epoch và bổ sung data augmentation. Đây cũng là kết luận quen thuộc trong các bài toán có tập huấn luyện nhỏ: tận dụng đặc trưng pretrained làm "mỏ neo" cho bộ phân loại nhẹ thường ổn định và hiệu quả hơn việc cố gắng huấn luyện sâu lại từ đầu.

## 7. Kết luận

Báo cáo đã thực hiện đầy đủ các bước trong yêu cầu của Bài 3: EDA, tiền xử lý, trích xuất đặc trưng deep, lưu file `.npy`, huấn luyện và so sánh ba bộ phân loại, đồng thời đối chiếu với một pipeline học sâu đầu-cuối. Kết quả tốt nhất đạt được với cấu hình ResNet50 + Linear SVM, accuracy khoảng 0,94 và F1-macro khoảng 0,93 trên tập kiểm tra. Pipeline VGG16 end-to-end (5 epoch, 4 lớp fine-tune) đạt accuracy khoảng 0,87 — thấp hơn pipeline truyền thống nhưng vẫn cho thấy backbone pretrained có thể được dùng theo cả hai hướng.

Một số hạn chế và hướng mở rộng. Tập dữ liệu chỉ có hai lớp; với bài toán nhiều nhãn cần thử thêm SVM phi tuyến (RBF) và một mạng phân loại nhỏ (MLP) trên đặc trưng CNN. Pipeline học sâu chưa dùng data augmentation, có thể cải thiện đáng kể khi có GPU và augmentation cơ bản (random flip, crop, color jitter). Cuối cùng, có thể thay backbone bằng Vision Transformer (ViT) hoặc Swin Transformer để khảo sát ảnh hưởng của transformer-based features so với CNN, đặc biệt khi tăng số mẫu lên hàng chục nghìn.

## 8. Tài nguyên tham khảo và mã nguồn

Toàn bộ mã nguồn, notebook thực nghiệm và các file liên quan của bài toán được lưu trữ trên GitHub nhằm phục vụ tái lập kết quả và tham khảo chi tiết quá trình triển khai.

GitHub Repository: https://github.com/ngtan369/Hybrid-Image-Classification

Google Colab Notebook: https://colab.research.google.com/github/ngtan369/Hybrid-Image-Classification/blob/main/notebooks/ex3_imageData.ipynb

Tập dữ liệu: Kaggle — `jcoral02/inriaperson` (kéo về tự động qua `kagglehub` trong notebook).

Repository bao gồm các thành phần chính:

- `notebooks/ex3_imageData.ipynb` — front-end Colab notebook chạy đầu-cuối.
- `modules/ml_utils.py` — module dataset discovery, EDA, classifier comparison, VOC bbox helpers, feature I/O.
- `modules/dl_utils.py` — module pretrained feature extraction và transfer-learning helpers.
- `features/` — vector đặc trưng `.npy` được sinh ra khi chạy notebook.
- `reports/report.pdf` — bản PDF của file báo cáo này.
- `README.md` — thông tin nhóm, môn học, GVHD, hướng dẫn chạy.

Google Colab được sử dụng làm môi trường thực thi chính để huấn luyện mô hình và chạy thực nghiệm; có thể chuyển đổi nhanh giữa CPU và GPU runtime tùy theo nhu cầu.

## 9. Phân công công việc

| Thành viên | MSSV | Email | Nhiệm vụ chính | Tỉ lệ đóng góp |
| ---------- | ---- | ----- | -------------- | -------------- |
| Nguyễn Trung Tân | xxxx | ngtan369@gmail.com | Pipeline, modules, notebook, báo cáo | 100% |

Bảng phân công thực tế của nhóm sẽ được điền lại trước khi nộp bài.
