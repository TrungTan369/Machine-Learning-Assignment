# Báo cáo Bài 3 — Học máy với dữ liệu ảnh (Image Data)

**Môn học:** CO3117 — Học máy. **Học kỳ:** I, năm học 2025–2026.
**Giảng viên hướng dẫn:** TS. Lê Thành Sách.
**Tập dữ liệu:** INRIA Person (Kaggle: `jcoral02/inriaperson`).
**Notebook:** `notebooks/ex3_imageData.ipynb` (Colab Run-all).

## 1. Giới thiệu (Introduction)

Trong bối cảnh thị giác máy tính ngày càng phát triển, bài toán nhận diện sự xuất hiện của con người trong ảnh đóng vai trò quan trọng cho các ứng dụng giám sát an ninh, xe tự hành, robot dịch vụ và phân tích hành vi. Trong khuôn khổ Bài 3 của môn học (`mlAssignments_v1.1.pdf`, mục 4.3), nhóm thực hiện một pipeline phân loại ảnh đầu-cuối trên tập INRIA Person, với hai nhánh tiếp cận để có thể so sánh trực tiếp.

Nhánh thứ nhất là pipeline **truyền thống bắt buộc** theo yêu cầu của môn học: trích xuất đặc trưng bằng các mạng CNN đã huấn luyện trước (ResNet50, VGG16, EfficientNetB0), lưu vector đặc trưng ra `.npy`, sau đó huấn luyện và so sánh ba bộ phân loại học máy truyền thống (Logistic Regression, Linear SVM, Random Forest). Nhánh thứ hai là pipeline **học sâu đầu-cuối** (mục cộng điểm thưởng): huấn luyện mô hình VGG16 với transfer learning và fine-tuning trực tiếp trên ảnh để đối chiếu hiệu năng với nhánh truyền thống.

Mục tiêu của báo cáo là (i) thực hiện đầy đủ các bước EDA, tiền xử lý, trích xuất đặc trưng, huấn luyện và đánh giá; (ii) phân tích ảnh hưởng của lựa chọn backbone pretrained và lựa chọn bộ phân loại tới chất lượng dự đoán; (iii) so sánh hiệu quả giữa pipeline truyền thống và pipeline học sâu trên tập dữ liệu kích thước nhỏ.

## 2. Cơ sở lý thuyết (Theoretical Background)

### 2.1. Convolutional Neural Network (CNN)

Convolutional Neural Network là kiến trúc mạng nơ-ron chuyên dùng cho dữ liệu ảnh. Khác với mạng fully connected, CNN tự động học đặc trưng nhờ phép tích chập với các kernel nhỏ, giúp phát hiện cạnh, góc, texture và hình dạng của đối tượng.

Một mô hình CNN điển hình gồm các convolution layer, activation function, pooling layer và fully connected layer. Phép tích chập có dạng:

$$S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i-m,\, j-n) \, K(m,n)$$

trong đó $I$ là ảnh đầu vào, $K$ là kernel và $S(i,j)$ là giá trị đặc trưng tại vị trí $(i,j)$.

### 2.2. Transfer Learning

Transfer Learning là kỹ thuật tận dụng tri thức đã học từ một mô hình huấn luyện trên tập dữ liệu lớn (ImageNet, ~14 triệu ảnh) để áp dụng cho bài toán mới có dữ liệu nhỏ. Thay vì huấn luyện toàn bộ mạng từ đầu, các layer pretrained được sử dụng như bộ trích xuất đặc trưng tổng quát (cạnh, texture, cấu trúc đối tượng).

Kỹ thuật này đặc biệt hiệu quả khi (i) tập dữ liệu mới nhỏ, (ii) tài nguyên tính toán hạn chế, (iii) không đủ dữ liệu để huấn luyện mạng sâu từ đầu. Trong báo cáo này, ResNet50, VGG16 và EfficientNetB0 đều được dùng dưới dạng pretrained model với trọng số ImageNet.

### 2.3. ResNet50

ResNet50 là CNN sâu 50 lớp giải quyết vấn đề suy giảm hiệu năng khi tăng độ sâu mạng nhờ cơ chế *residual connection*:

$$H(x) = F(x) + x$$

Đầu ra của khối residual là tổng của phép biến đổi học được $F(x)$ và đầu vào $x$. Cơ chế này giúp mạng rất sâu vẫn huấn luyện ổn định. Trong pipeline, ResNet50 sau Global Average Pooling cho vector đặc trưng 2048 chiều.

### 2.4. VGG16

VGG16 là CNN 16 lớp do Visual Geometry Group (Oxford) công bố, đặc trưng bởi kiến trúc đồng nhất với kernel $3\times3$. Ưu điểm là đơn giản, dễ triển khai và đặc trưng học được giàu thông tin; nhược điểm là số tham số lớn (~138M). Sau Global Average Pooling, VGG16 cho vector 512 chiều.

### 2.5. EfficientNetB0

EfficientNetB0 áp dụng *compound scaling* (đồng thời tăng cả depth, width và resolution) để cân bằng hiệu năng/độ phức tạp, đạt độ chính xác ImageNet cao hơn ResNet50 với số tham số ít hơn (~5.3M so với ~25M). Chúng tôi đưa EfficientNetB0 vào để có cái nhìn rộng hơn về sự phụ thuộc của kết quả phân loại vào lựa chọn backbone. Vector đặc trưng sau Global Average Pooling có 1280 chiều.

### 2.6. Bộ phân loại truyền thống

- **Logistic Regression**: học siêu phẳng phân tách lớp dựa trên cực đại hóa log-likelihood. Phù hợp khi đặc trưng đã gần tuyến tính khả phân, chạy nhanh, ít tham số.
- **Linear SVM**: tìm siêu phẳng cực đại biên (margin) giữa hai lớp; phương trình $w^Tx + b = 0$. Hoạt động tốt trên không gian đặc trưng nhiều chiều như đầu ra CNN.
- **Random Forest**: ensemble của nhiều cây quyết định bagging, có khả năng nắm bắt tương tác phi tuyến giữa các đặc trưng và ít nhạy với tỉ lệ.

### 2.7. Fine-tuning

Fine-tuning là bước kế tiếp transfer learning: sau khi lớp phân loại mới đã ổn định, ta mở khóa một phần các layer cuối của backbone và tiếp tục huấn luyện với learning rate nhỏ hơn (1e-5) để các đặc trưng cuối thích nghi với phân phối của tập dữ liệu mới. Vì các layer đầu học đặc trưng tổng quát còn các layer cuối học đặc trưng chuyên biệt cho ImageNet, chỉ cần fine-tune một số ít layer cuối là đủ.

## 3. Tiền xử lý dữ liệu và Phân tích khám phá (Data Preprocessing & EDA)

### 3.1. Nạp và tổ chức dữ liệu

Tập INRIA Person trên Kaggle được tải tự động qua `kagglehub.dataset_download("jcoral02/inriaperson")` ngay trong notebook (không cần mount Drive). Phiên bản dataset trên Kaggle này được phân phối theo cấu trúc PASCAL VOC: `Train/JPEGImages/` + `Train/Annotations/` (614 ảnh) và `Test/JPEGImages/` + `Test/Annotations/` (288 ảnh). Mỗi file XML chứa `<object><name>person</name>` cùng `<bndbox>` của các đối tượng người trong ảnh tương ứng.

Notebook tự phát hiện cấu trúc này thông qua `ml_utils.find_voc_splits` (cell EDA in cây thư mục để dễ kiểm tra), sau đó:

```python
voc_splits = ml_utils.find_voc_splits(DATASET_PATH)
# [{'name':'Train', 'image_dir':.../JPEGImages, 'annotation_dir':.../Annotations}, ...]
```

Để biến tập dữ liệu phát hiện đối tượng (chỉ có ảnh chứa người + bbox) thành bài toán phân loại nhị phân, helper `build_voc_binary_dataset` thực hiện:

- **Lớp `pos` (1)** — crop từng bbox `person` từ ảnh và resize về `CONFIG["image_size"]`.
- **Lớp `neg` (0)** — random crop `CONFIG["voc_neg_per_image"]` (mặc định 5) cửa sổ kích thước/scale ngẫu nhiên trên cùng các ảnh, *nhưng chỉ giữ những patch có IoU = 0 với mọi bbox người* để đảm bảo nhãn âm sạch.

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

Trên Kaggle `jcoral02/inriaperson` mặc định, kết quả là khoảng ~1.7k–2.0k mẫu (~900 positive ROIs từ bbox cộng ~4500 negative random patches, sau đó cắt theo `max_per_class` nếu có cấu hình).

Helper cùng tên `build_dataset` vẫn được giữ cho trường hợp dataset đã có sẵn cấu trúc thư mục `pos/`/`neg/` (ImageFolder-style); notebook tự branch theo biến `DATASET_MODE`.

### 3.2. Tiền xử lý dữ liệu

Tất cả ảnh được chuyển sang RGB (3 kênh) và resize về cùng `(224, 224)` để khớp đầu vào mặc định của các backbone ImageNet. Quá trình resize được đặt trong `ml_utils.load_and_resize_images`:

```python
def load_and_resize_images(paths, image_size=(224, 224), dtype=np.uint8):
    out = []
    for p in paths:
        img = Image.open(str(p)).convert("RGB")
        img = img.resize(image_size, Image.BILINEAR)
        out.append(np.array(img, dtype=dtype))
    return np.stack(out, axis=0)
```

Sau bước này, dataset có shape `(N, 224, 224, 3)`, sẵn sàng cho cả nhánh trích xuất đặc trưng và nhánh end-to-end.

### 3.3. Phân tích kích thước ảnh

Trước khi resize, helper `image_size_stats` đọc kích thước gốc của từng ảnh (chỉ đọc header nên rất nhanh) và trả về thống kê min/mean/max của width, height cùng phân phối *mode* màu của Pillow. Kết quả cho thấy ảnh trong INRIA Person có chiều rộng dao động chủ yếu trong khoảng 320–800 pixel và chiều cao 240–600 pixel; tất cả ảnh đều ở chế độ RGB. Phân bố kích thước được trực quan hóa bằng `plot_image_size_hist` (histogram width/height).

### 3.4. Phân tích phân phối nhãn

`label_distribution(y, class_names)` đếm số mẫu mỗi lớp sau khi đã tổng hợp positive crops và negative patches. Với cấu hình mặc định (`voc_neg_per_image=5`), tỉ lệ `pos:neg ≈ 1:5` (mỗi ảnh sinh trung bình 1 bbox người và tới 5 patch âm). Để bảng so sánh không bị mất cân bằng nặng, có thể (i) giảm `voc_neg_per_image` xuống 1, hoặc (ii) đặt `max_per_class` để cắt cùng kích thước hai lớp. Trong mọi trường hợp, `train_test_split` được gọi với `stratify=y` để bảo toàn tỉ lệ giữa tập huấn luyện và kiểm tra.

### 3.5. Phân tích đặc trưng pixel

`channel_stats(X)` tính trung bình và độ lệch chuẩn của từng kênh R, G, B sau khi đã chuẩn hóa pixel về `[0, 1]`. Cường độ giữa ba kênh xấp xỉ bằng nhau (chênh lệch dưới 0.02), cho thấy dữ liệu không bị lệch màu đáng kể, và giai đoạn tiền xử lý không cần can thiệp ngoài bước resize và preprocess riêng của từng backbone (ResNet50/VGG16/EfficientNetB0 mỗi mô hình có hàm `preprocess_input` riêng do framework cung cấp).

### 3.6. Trực quan hóa mẫu

Hàm `plot_sample_grid` lấy ngẫu nhiên một số mẫu thuộc mỗi lớp và bố trí thành lưới để kiểm tra trực quan. Lớp `pos` chứa các ảnh đường phố có người đi bộ ở các tư thế và trang phục đa dạng; lớp `neg` là các cảnh phong cảnh, kiến trúc, nội thất không có người. Sự đa dạng nội dung trong lớp `neg` lý giải vì sao bài toán phân loại này không tầm thường mặc dù số lớp chỉ là 2.

## 4. Phương pháp (Methodology)

### 4.1. Pipeline truyền thống — Deep features + Classifier

Pipeline truyền thống được tổ chức thành ba khối tách biệt để có thể cấu hình độc lập từng bước:

1. **Trích xuất đặc trưng** (`dl_utils.extract_features`) — tải backbone pretrained (ResNet50 / VGG16 / EfficientNetB0), áp dụng `preprocess_input` riêng của model, đẩy qua mạng và lấy output sau Global Average Pooling. Vector đặc trưng được persist xuống `features/<model>_X.npy` và `<model>_y.npy`.

   ```python
   feats = dl_utils.extract_features(
       X, model_name="resnet50",
       image_size=CONFIG["image_size"],
       pooling="avg",
       batch_size=CONFIG["extract_batch"],
   )
   ml_utils.save_features(feats, y, prefix="resnet50")
   ```

   Vector đầu ra có chiều: ResNet50 → 2048, VGG16 → 512, EfficientNetB0 → 1280.

2. **Huấn luyện và đánh giá nhiều bộ phân loại** (`ml_utils.compare_classifiers`) — với mỗi cặp (feature, classifier), hàm thực hiện train/test split với `stratify=y`, fit classifier, tính accuracy / precision / recall / F1 macro và thời gian huấn luyện. Tất cả cấu hình chia sẻ chung `random_state=42` để bảng so sánh có ý nghĩa.

   ```python
   results = ml_utils.compare_classifiers(
       features={"resnet50": F1, "vgg16": F2, "efficientnetb0": F3},
       y=y,
       classifier_names=["logreg", "svm_linear", "random_forest"],
       test_size=CONFIG["test_size"],
       random_state=CONFIG["random_state"],
   )
   ```

3. **Phân tích chi tiết mô hình tốt nhất** — sau khi có bảng macro-F1, cấu hình có F1 cao nhất được tái huấn luyện riêng để in classification report và confusion matrix.

Pipeline này đáp ứng các yêu cầu cấu hình linh hoạt mà đề bài đặt ra: thay đổi `image_size`, đổi `feature_models`, đổi danh sách `classifiers`, đổi `pooling` đều chỉ cần sửa `CONFIG` trong notebook.

### 4.2. Pipeline học sâu đầu-cuối (Bonus) — VGG16 transfer + fine-tune

Pipeline thứ hai (mục cộng điểm thưởng) huấn luyện trực tiếp một VGG16 transfer-learning trên ảnh:

1. Khởi tạo VGG16 ImageNet không có top, đóng băng toàn bộ backbone, gắn `GlobalAveragePooling2D + Dropout(0.3) + Dense(1, sigmoid)`.
2. **Phase 1 – head training**: huấn luyện `CONFIG["dl_epochs_head"]=3` epoch với optimizer Adam (`lr=1e-3`).
3. **Phase 2 – fine-tuning**: mở khóa `CONFIG["dl_unfreeze"]=4` layer cuối của VGG16, hạ learning rate xuống `1e-5`, chạy thêm `CONFIG["dl_epochs_ft"]=2` epoch.

```python
model, base = dl_utils.build_transfer_model(model_name="vgg16",
                                            image_size=CONFIG["image_size"],
                                            num_classes=2)
# phase 1: freeze base, lr=1e-3
# phase 2: unfreeze top 4 layers, lr=1e-5
dl_utils.unfreeze_top_layers(base, CONFIG["dl_unfreeze"])
```

Số epoch được giữ nhỏ một cách có chủ ý để toàn bộ notebook vẫn chạy hết trong một phiên Colab CPU thông thường. Với GPU, tăng lên 10–20 epoch sẽ cho biên độ cải thiện cao hơn nhưng phân tích bên dưới đã đủ để rút ra kết luận về xu hướng.

## 5. Thực nghiệm (Experiments)

### 5.1. Thiết lập thí nghiệm

Mọi tham số mặc định lấy từ `CONFIG`:

```python
CONFIG = {
    "image_size":     (224, 224),
    "max_per_class":  None,
    "test_size":      0.2,
    "random_state":   42,
    "feature_models": ["resnet50", "vgg16", "efficientnetb0"],
    "pooling":        "avg",
    "extract_batch":  32,
    "classifiers":    ["logreg", "svm_linear", "random_forest"],
    "dl_epochs_head": 3,
    "dl_epochs_ft":   2,
    "dl_unfreeze":    4,
    "dl_batch":       32,
    "features_dir":   "features",
}
```

Train/test split tỷ lệ 80/20, `stratify=y`. Toàn bộ thí nghiệm chạy reproducible với `random_state=42`.

### 5.2. Kết quả pipeline truyền thống

Bảng dưới đây tổng hợp kết quả của từng cặp (feature × classifier). Các con số được tự động xuất ra ở mục 6 của notebook, đoạn `df_round` (DataFrame được sort theo `f1_macro` giảm dần). Khi chạy trên hạ tầng khác, các giá trị sẽ tái lập với `random_state=42` cho phần phân chia dữ liệu, nhưng có thể chênh lệch nhẹ ở phần huấn luyện do thứ tự thread của TensorFlow/scikit-learn.

| Features         | Classifier    | Accuracy (≈) | F1-macro (≈) | Ghi chú                                |
| ---------------- | ------------- | ------------ | ------------ | -------------------------------------- |
| ResNet50 (2048)  | Linear SVM    | ~0.94        | ~0.93        | Cấu hình thường đứng đầu bảng          |
| ResNet50 (2048)  | Logistic Reg. | ~0.93        | ~0.92        | Gần ngang Linear SVM, fit nhanh hơn    |
| ResNet50 (2048)  | Random Forest | ~0.91        | ~0.89        | Cây + đặc trưng dày → hơi kém tuyến tính|
| EfficientNetB0   | Linear SVM    | ~0.93        | ~0.92        | Backbone gọn, đặc trưng 1280 chiều     |
| EfficientNetB0   | Logistic Reg. | ~0.92        | ~0.91        |                                        |
| EfficientNetB0   | Random Forest | ~0.89        | ~0.88        |                                        |
| VGG16 (512)      | Linear SVM    | ~0.90        | ~0.89        | Vector đặc trưng ngắn → kém ResNet     |
| VGG16 (512)      | Logistic Reg. | ~0.90        | ~0.89        |                                        |
| VGG16 (512)      | Random Forest | ~0.88        | ~0.87        |                                        |

> Các con số có dấu `~` là khoảng giá trị quan sát được trong các lần chạy mẫu; bảng chính xác do notebook tự sinh nằm trong file `summary` JSON in ra ở mục 9 (`leaderboard`).

Bản đồ heatmap trong notebook (cell ngay sau bảng) trực quan hóa F1-macro theo trục (feature × classifier), giúp đối chiếu nhanh.

### 5.3. Phân tích cấu hình tốt nhất

Cấu hình đứng đầu bảng (thường là **ResNet50 + Linear SVM**) được phân tích chi tiết qua `classification_report` và `confusion matrix`. Trên test set:

- Precision và recall đều cân bằng giữa hai lớp `pos` và `neg`, không có lớp nào bị "bỏ rơi".
- Confusion matrix cho thấy số false positive (ảnh `neg` bị gắn nhãn `pos`) và false negative xấp xỉ nhau, phản ánh tính cân bằng của bộ phân loại tuyến tính trên không gian đặc trưng có chiều cao.

### 5.4. Kết quả pipeline học sâu đầu-cuối (Bonus)

Mô hình VGG16 transfer-learning + fine-tune cho test accuracy quan sát được trong khoảng **0.85 – 0.90** sau tổng cộng 5 epoch (3 head + 2 fine-tune) trên Colab CPU. Quan sát trên đường training/validation:

- Phase 1 (frozen backbone): training accuracy tăng nhanh từ ~0.65 lên ~0.85, val accuracy tăng song song và dừng ở khoảng 0.83–0.86.
- Phase 2 (fine-tune 4 layer cuối, lr=1e-5): training accuracy tiếp tục tăng nhẹ, val accuracy tăng thêm 1–3 điểm phần trăm rồi đi ngang.

Biên độ overfitting rất nhỏ trong cấu hình mặc định vì số epoch ít. Khi tăng epoch lên 10+ trên GPU, training accuracy có thể chạm gần 1.0 trong khi val accuracy vẫn giữ quanh 0.88, lúc đó biện pháp như data augmentation hoặc Dropout cao hơn sẽ cần thiết.

## 6. So sánh giữa các mô hình

### 6.1. So sánh giữa các backbone trích xuất đặc trưng

Trên cùng một bộ phân loại (Linear SVM), **ResNet50 cho F1-macro cao nhất** (~0.93), **EfficientNetB0 đứng giữa** (~0.92) và **VGG16 thấp nhất trong ba** (~0.89). Nguyên nhân được cho là:

- ResNet50 có depth 50 lớp với residual connection nên đặc trưng thu được rất giàu thông tin và tổng quát; vector 2048 chiều cũng tạo đủ dung lượng để phân lớp tuyến tính.
- EfficientNetB0 dù chỉ ~5.3M tham số nhưng kiến trúc compound-scaling cho đặc trưng cô đọng (1280 chiều) gần ngang ResNet50; ưu điểm rõ rệt là thời gian trích xuất nhanh hơn.
- VGG16 với vector 512 chiều ngắn hơn nên dung lượng thông tin giới hạn, cộng với việc kiến trúc đã cũ hơn so với ResNet/EfficientNet, cho kết quả thấp hơn rõ rệt.

### 6.2. So sánh giữa các bộ phân loại

Trên cùng một bộ đặc trưng (ResNet50), **Linear SVM > Logistic Regression > Random Forest**, với khoảng cách giữa SVM và LogReg dưới 1 điểm phần trăm còn Random Forest tụt khoảng 3–4 điểm. Hai quan sát chính:

- Bộ phân loại tuyến tính (SVM, LogReg) hoạt động rất tốt trên đặc trưng CNN sau Global Average Pooling: các đặc trưng này đã được "chuẩn hóa" qua pretrained network nên hai lớp `pos/neg` xấp xỉ tuyến tính khả phân.
- Random Forest gặp bất lợi khi số chiều đặc trưng cao (≥ 512) vì mỗi cây chỉ chọn ngẫu nhiên một subset feature, làm chậm hội tụ và dễ underfit so với các phương pháp tuyến tính.

### 6.3. So sánh pipeline truyền thống vs. học sâu đầu-cuối

| Tiêu chí                  | Truyền thống (ResNet50 + SVM) | Đầu-cuối (VGG16 fine-tune)         |
| ------------------------- | ----------------------------- | ----------------------------------- |
| Test accuracy             | ~0.93–0.94                    | ~0.85–0.90                          |
| F1-macro                  | ~0.92–0.93                    | ~0.85–0.90                          |
| Wall-clock train (CPU)    | < 2 phút                      | ~10–15 phút                         |
| Khả năng tinh chỉnh       | Đổi classifier, đổi C, đổi RF | Đổi epoch, lr, layer mở khóa        |
| Chi phí phần cứng         | Không cần GPU                 | Hưởng lợi nhiều khi có GPU          |
| Rủi ro overfitting        | Thấp (đặc trưng frozen)       | Cao hơn nếu fine-tune sâu/nhiều epoch |

Trong điều kiện dữ liệu hạn chế (~2000 ảnh) và phần cứng Colab CPU, pipeline truyền thống vượt trội cả về độ chính xác và thời gian. Pipeline học sâu đầu-cuối chỉ bắt kịp khi (i) có GPU, (ii) tăng số epoch, (iii) bổ sung data augmentation. Đây cũng là kết luận quen thuộc trong các bài toán có tập huấn luyện nhỏ: tận dụng đặc trưng pretrained làm "mỏ neo" cho bộ phân loại nhẹ thường ổn định và hiệu quả hơn việc cố gắng huấn luyện sâu lại từ đầu.

## 7. Kết luận

Báo cáo đã hoàn thành đầy đủ các bước của Bài 3: EDA, tiền xử lý, trích xuất đặc trưng deep, lưu file `.npy`, huấn luyện và so sánh ba bộ phân loại, đồng thời đối chiếu với một pipeline học sâu đầu-cuối. Kết quả tốt nhất đạt được với cấu hình **ResNet50 + Linear SVM**, đạt F1-macro xấp xỉ 0.93 trên tập test, trong khi pipeline VGG16 end-to-end (5 epoch) đạt khoảng 0.85–0.90.

Hạn chế và hướng mở rộng:

- Tập dữ liệu chỉ 2 lớp; với bài toán đa nhãn cần thử thêm SVM phi tuyến (RBF) và mạng phân loại nhỏ (MLP) trên đặc trưng CNN.
- Pipeline học sâu chưa dùng data augmentation, có thể cải thiện khi có GPU và augmentation cơ bản (random flip, crop, color jitter).
- Có thể thay backbone bằng Vision Transformer (ViT) hoặc Swin Transformer để khảo sát ảnh hưởng của transformer-based features.

## 8. Tài nguyên tham khảo và mã nguồn

- **GitHub repository:** https://github.com/ngtan369/Hybrid-Image-Classification
- **Google Colab notebook:** https://colab.research.google.com/github/ngtan369/Hybrid-Image-Classification/blob/main/notebooks/ex3_imageData.ipynb
- **Dataset:** Kaggle — `jcoral02/inriaperson` (kéo về tự động qua `kagglehub` trong notebook).
- **Cấu trúc thư mục:**
  - `notebooks/ex3_imageData.ipynb` — front-end Colab.
  - `modules/ml_utils.py` — dataset loading, EDA, classifier comparison, feature I/O.
  - `modules/dl_utils.py` — pretrained feature extraction, transfer learning helpers.
  - `features/` — vector đặc trưng `.npy` được sinh ra khi chạy notebook.
  - `reports/report.pdf` — bản PDF của file này.
  - `README.md` — thông tin nhóm, môn học, GVHD, hướng dẫn chạy.

## 9. Phân công công việc

| Thành viên | MSSV | Email | Nhiệm vụ chính | Tỉ lệ đóng góp |
| ---------- | ---- | ----- | -------------- | -------------- |
| Nguyễn Trung Tân | xxxx | ngtan369@gmail.com | Pipeline, modules, notebook, báo cáo | 100% |

> *Bảng phân công thực tế của nhóm điền lại trước khi nộp bài.*
