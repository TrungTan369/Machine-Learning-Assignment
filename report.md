1. Giới thiệu (Introduction)
Trong bối cảnh thị giác máy tính (Computer Vision) ngày càng phát triển, bài toán phát hiện con người trong ảnh (human detection) đóng vai trò quan trọng trong nhiều ứng dụng thực tiễn như giám sát an ninh, xe tự hành, robot dịch vụ và phân tích hành vi. Khác với bài toán phân loại ảnh thông thường, mục tiêu của bài toán này không chỉ là xác định sự xuất hiện của con người mà còn phải xác định vị trí của người trong ảnh thông qua bounding box.
Trong báo cáo này, tập dữ liệu INRIA Person được sử dụng để xây dựng hệ thống phát hiện người trong ảnh. Bộ dữ liệu bao gồm các ảnh chứa người cùng với thông tin vị trí đối tượng, phục vụ cho bài toán object detection. Do kích thước tập dữ liệu tương đối nhỏ, việc huấn luyện trực tiếp một mô hình học sâu từ đầu có thể gặp khó khăn về khả năng tổng quát hóa.
Để giải quyết vấn đề này, hai hướng tiếp cận được áp dụng và so sánh. Thứ nhất là phương pháp hybrid, trong đó mô hình học sâu pretrained như ResNet50 được sử dụng để trích xuất đặc trưng từ các vùng ảnh, sau đó thuật toán học máy truyền thống như SVM được dùng để xác định xem vùng ảnh đó có chứa người hay không. Các vùng được xác định thông qua sliding window hoặc region proposal để sinh ra bounding box cho đối tượng.
Thứ hai là phương pháp học sâu end-to-end sử dụng kiến trúc CNN pretrained như VGG16 kết hợp transfer learning và fine-tuning nhằm học trực tiếp đặc trưng của đối tượng người từ dữ liệu ảnh.
Mục tiêu của báo cáo là xây dựng pipeline phát hiện người trong ảnh, phân tích hiệu quả của các phương pháp khác nhau và đánh giá khả năng định vị đối tượng thông qua bounding box.
2. Cơ sở lý thuyết (Theoretical Background)
2.1. Convolutional Neural Network (CNN)
Convolutional Neural Network (CNN) là một kiến trúc mạng nơ-ron chuyên dùng cho các bài toán xử lý ảnh. Khác với mạng neural truyền thống, CNN có khả năng tự động học đặc trưng từ dữ liệu ảnh thông qua các lớp convolution. Các lớp này hoạt động bằng cách áp dụng các kernel nhỏ lên ảnh đầu vào để phát hiện các đặc trưng như cạnh, góc, texture hoặc hình dạng của đối tượng.
Một mô hình CNN điển hình thường bao gồm các thành phần chính như convolution layer, activation function, pooling layer và fully connected layer. Trong đó, convolution layer đóng vai trò quan trọng nhất vì đây là nơi mô hình học ra các đặc trưng của ảnh.
Phép tích chập trong CNN có thể được biểu diễn như sau:
$S(i,j)=(I*K)(i,j)=\sum_m\sum_n I(i-m,j-n)K(m,n)$
Trong đó:
$I$ là ảnh đầu vào
$K$ là kernel (bộ lọc)
$S(i,j)$ là giá trị đặc trưng tại vị trí $((i,j))$
Nhờ khả năng tự động học đặc trưng, CNN đã trở thành nền tảng của hầu hết các mô hình hiện đại trong Computer Vision.
2.2. Transfer Learning
Transfer Learning là kỹ thuật sử dụng lại tri thức từ các mô hình đã được huấn luyện trước trên tập dữ liệu lớn để áp dụng vào bài toán mới. Thay vì huấn luyện toàn bộ mô hình từ đầu, các mô hình pretrained có thể được tận dụng như bộ trích xuất đặc trưng mạnh mẽ.
Trong các bài toán xử lý ảnh, các mô hình như VGG16 hoặc ResNet50 thường được huấn luyện trước trên ImageNet với hàng triệu ảnh thuộc hàng nghìn lớp khác nhau. Các mô hình này đã học được nhiều đặc trưng tổng quát của ảnh như cạnh, texture và cấu trúc đối tượng.
Kỹ thuật Transfer Learning đặc biệt hiệu quả khi:
Tập dữ liệu mới có kích thước nhỏ
Tài nguyên tính toán hạn chế
Không đủ dữ liệu để huấn luyện mạng sâu từ đầu
Trong bài toán thực nghiệm, cả ResNet50 và VGG16 đều được sử dụng dưới dạng pretrained model với trọng số được huấn luyện trên ImageNet.
2.3. ResNet50
ResNet50 là một kiến trúc CNN sâu gồm 50 lớp, được giới thiệu nhằm giải quyết vấn đề suy giảm hiệu năng khi số lớp của mạng tăng lên quá lớn. Điểm đặc trưng của ResNet là cơ chế residual connection, cho phép dữ liệu được truyền trực tiếp qua nhiều lớp mà không bị mất thông tin quan trọng.
Residual block trong ResNet được biểu diễn như sau:
$H(x)=F(x)+x$
Trong đó:
$x$ là đầu vào
$F(x)$ là phần biến đổi học được bởi mạng
$H(x)$ là đầu ra cuối cùng
Cơ chế này giúp ResNet50 có thể huấn luyện các mạng rất sâu mà vẫn duy trì hiệu quả cao. Trong bài toán này, ResNet50 được sử dụng làm bộ trích xuất đặc trưng từ ảnh trước khi đưa vào mô hình SVM để phân loại.
2.4. VGG16
VGG16 là một kiến trúc CNN nổi tiếng được phát triển bởi nhóm Visual Geometry Group của Đại học Oxford. Mô hình gồm 16 lớp học được và sử dụng các convolution kernel kích thước nhỏ (3 \times 3).
Đặc điểm nổi bật của VGG16 là kiến trúc đơn giản, đồng nhất và dễ triển khai. Mặc dù số lượng tham số lớn hơn nhiều mô hình hiện đại khác, VGG16 vẫn được sử dụng rộng rãi trong các bài toán transfer learning nhờ khả năng trích xuất đặc trưng hiệu quả.
Trong bài toán thực nghiệm, VGG16 được sử dụng theo hướng end-to-end. Phần convolutional base của mô hình được giữ lại để tận dụng các đặc trưng học từ ImageNet, trong khi lớp fully connected cuối cùng được thay đổi để phù hợp với bài toán phân loại nhị phân.
2.5. Support Vector Machine (SVM)
Support Vector Machine là một thuật toán Machine Learning phổ biến cho các bài toán phân loại. Ý tưởng chính của SVM là tìm ra một siêu phẳng tối ưu nhằm phân tách các lớp dữ liệu với khoảng cách lớn nhất.
Phương trình siêu phẳng của SVM được biểu diễn như sau:
$w^Tx+b=0$
Trong đó:
$w$ là vector trọng số
$b$ là bias của mô hình
SVM hoạt động đặc biệt hiệu quả trong không gian đặc trưng có số chiều lớn. Khi kết hợp với các đặc trưng được trích xuất từ ResNet50, SVM có thể đạt hiệu quả phân loại tốt ngay cả khi kích thước tập dữ liệu không lớn.
Trong bài thực nghiệm, SVM với linear kernel được sử dụng để thực hiện phân loại trên vector đặc trưng được sinh ra từ ResNet50.
2.6. Fine-tuning
Fine-tuning là kỹ thuật tiếp tục huấn luyện một phần hoặc toàn bộ mô hình pretrained trên tập dữ liệu mới nhằm điều chỉnh các đặc trưng phù hợp hơn với bài toán cụ thể.
Thông thường, các lớp đầu của CNN học các đặc trưng tổng quát như cạnh hoặc texture, trong khi các lớp sâu hơn học các đặc trưng chuyên biệt hơn cho từng bài toán. Vì vậy, trong quá trình fine-tuning, chỉ một số lớp cuối của mạng thường được mở khóa để tiếp tục huấn luyện.
Trong bài toán này, sau giai đoạn transfer learning ban đầu, một phần các lớp cuối của VGG16 được mở khóa và huấn luyện tiếp với learning rate nhỏ hơn. Mục tiêu của quá trình này là giúp mô hình thích nghi tốt hơn với dữ liệu INRIA Person và cải thiện hiệu năng phân loại.
3. Tiền xử lý dữ liệu và Phân tích khám phá (Data Preprocessing & EDA)
Trong nghiên cứu này, tập dữ liệu INRIA Person được sử dụng để phục vụ cho bài toán nhận diện sự xuất hiện của con người trong ảnh. Dữ liệu được tải từ Kaggle và giữ nguyên cấu trúc thư mục ban đầu. Quá trình tiền xử lý và phân tích dữ liệu được thực hiện nhằm đảm bảo dữ liệu đầu vào có định dạng phù hợp và hiểu rõ đặc điểm của tập dữ liệu trước khi huấn luyện mô hình.
3.1. Nạp và tổ chức dữ liệu
Dữ liệu ảnh được thu thập bằng cách duyệt toàn bộ các thư mục con và lấy đường dẫn đến các file ảnh thông qua hàm `load_image_paths` trong module `ml_utils`:
```python
class_names = sorted([d.name for d in Path(dataset_for_loader).iterdir() if d.is_dir()])

X_list = []
y_list = []

for i, cls in enumerate(class_names):
    paths = load_image_paths(Path(dataset_for_loader) / cls)
    imgs = load_and_resize_images(paths, image_size=image_size)
    X_list.append(imgs)
    y_list.append(np.full(len(imgs), i))

X_raw = np.vstack(X_list)
y_raw = np.concatenate(y_list)
```
Trong đó, mỗi thư mục con được xem như một lớp (class) và được gán nhãn số tương ứng. Kết quả thu được là tập dữ liệu ảnh `X_raw` và nhãn `y_raw`.
3.2. Tiền xử lý dữ liệu
Tất cả các ảnh được chuyển đổi về cùng một kích thước chuẩn là (224 \times 224) pixel nhằm đảm bảo tính nhất quán khi đưa vào mô hình. Đồng thời, ảnh được chuyển về định dạng RGB với 3 kênh màu. Quá trình này được thực hiện thông qua hàm `load_and_resize_images`:
```python
def load_and_resize_images(paths, image_size=(128,128)):
    imgs = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        img = img.resize(image_size)
        imgs.append(np.array(img))
    return np.stack(imgs, axis=0)
```
Sau bước này, mỗi ảnh được biểu diễn dưới dạng một tensor có kích thước (224 x 224 x 3), phù hợp cho cả mô hình học máy và học sâu.
3.3. Phân tích phân phối dữ liệu
Sau khi nạp dữ liệu, phân phối số lượng ảnh theo từng lớp được thống kê như sau:
```python
unique, counts = np.unique(y_raw, return_counts=True)
label_dist = dict(zip([class_names[i] for i in unique], counts))

for cls, count in label_dist.items():
    print(f"{cls}: {count} images")
```
Kết quả cho thấy tập dữ liệu gồm tổng cộng 902 ảnh, trong đó:
Train: 614 ảnh
Test: 288 ảnh
Phân phối này không cân bằng, với tỷ lệ xấp xỉ 2:1 giữa hai nhóm. Tuy nhiên, cần lưu ý rằng hai lớp này thực chất phản ánh cách chia dữ liệu thành tập huấn luyện và kiểm tra, chứ không phải là nhãn phân loại mang ý nghĩa ngữ nghĩa như “có người” hay “không có người”. Do đó, việc sử dụng trực tiếp cấu trúc này như nhãn đầu ra có thể ảnh hưởng đến khả năng học đúng bản chất bài toán của mô hình.
3.4. Phân tích đặc trưng ảnh
Ngoài phân phối dữ liệu, một số đặc trưng cơ bản của ảnh cũng được phân tích. Cụ thể, giá trị trung bình của các kênh màu RGB được tính toán như sau:
```python
ch_means = X_raw.mean(axis=(0, 1, 2))
plt.bar(['Red', 'Green', 'Blue'], ch_means)
```
Kết quả cho thấy cường độ pixel giữa ba kênh màu tương đối đồng đều, cho thấy dữ liệu không bị lệch màu đáng kể. Điều này giúp giảm nhu cầu áp dụng các kỹ thuật chuẩn hóa phức tạp trong giai đoạn tiền xử lý.
3.5. Minh họa phân phối dữ liệu
Phân phối số lượng ảnh theo từng lớp được trực quan hóa bằng biểu đồ cột:
```python
sns.barplot(x=list(label_dist.keys()), y=list(label_dist.values()))
plt.title("Label Distribution")
```
![alt text](image1.png)
Biểu đồ cho thấy sự chênh lệch rõ rệt giữa hai nhóm dữ liệu, phù hợp với kết quả thống kê đã trình bày ở trên.
Quá trình tiền xử lý đã đảm bảo tất cả các ảnh được đưa về cùng định dạng và kích thước, sẵn sàng cho các bước trích xuất đặc trưng và huấn luyện mô hình. Tuy nhiên, một hạn chế quan trọng là cấu trúc nhãn hiện tại chưa phản ánh đúng mục tiêu của bài toán (phân loại “có người” và “không có người”), mà chỉ dựa trên cách chia dữ liệu thành tập huấn luyện và kiểm tra. Điều này cần được xem xét và cải thiện trong các bước tiếp theo để nâng cao hiệu quả của mô hình.
4. Phương pháp (Methodology)
Trong nghiên cứu này, hai phương pháp được áp dụng để giải quyết bài toán nhận diện sự xuất hiện của con người trong ảnh, bao gồm phương pháp hybrid (kết hợp học sâu và học máy truyền thống) và phương pháp học sâu end-to-end. Hai hướng tiếp cận này được triển khai song song nhằm đánh giá hiệu quả của việc sử dụng đặc trưng trích xuất sẵn so với việc huấn luyện trực tiếp trên dữ liệu ảnh.
4.1. Phương pháp Hybrid (Deep Feature Extraction + Machine Learning)
Trong phương pháp này, mô hình học sâu không được sử dụng để phân loại trực tiếp mà đóng vai trò như một bộ trích xuất đặc trưng. Cụ thể, kiến trúc ResNet50 được sử dụng để chuyển đổi ảnh đầu vào thành vector đặc trưng có kích thước cố định.
Quá trình trích xuất đặc trưng được thực hiện thông qua hàm `extract_features_pretrained` trong module `dl_utils`. Hàm này khởi tạo mô hình pretrained tương ứng (ResNet50 hoặc VGG16), áp dụng hàm tiền xử lý phù hợp, và thực hiện suy diễn (inference) để thu được vector đặc trưng:
```python
def extract_features_pretrained(X_images, model_name='resnet50', batch_size=32, pooling='avg', verbose=1):
    H,W = X_images.shape[1], X_images.shape[2]
    model, preprocess = _get_model_and_preprocess(model_name, (H,W,3), pooling)
    X_proc = preprocess(X_images.copy())
    feats = model.predict(X_proc, batch_size=batch_size, verbose=verbose)
    return feats
```
Sau bước này, mỗi ảnh được biểu diễn bởi một vector đặc trưng 2048 chiều (đối với ResNet50 với Global Average Pooling):
```python
X_features = extract_features_pretrained(
    X_raw,
    model_name=model_name,
    batch_size=CONFIG['batch_size'],
    pooling='avg'
)
```
Tiếp theo, tập dữ liệu được chia thành tập huấn luyện và kiểm tra với tỷ lệ 80/20:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_raw, test_size=0.2, random_state=42
)
```
Cuối cùng, mô hình SVM với kernel tuyến tính được sử dụng để huấn luyện trên không gian đặc trưng:
```python
clf = SVC(kernel='linear', C=1.0, probability=True)
clf.fit(X_train, y_train)
```
Phương pháp này tận dụng khả năng học đặc trưng mạnh mẽ của mô hình học sâu đã được huấn luyện trước, đồng thời sử dụng thuật toán học máy truyền thống để thực hiện phân loại trên tập dữ liệu có kích thước hạn chế.
4.2. Phương pháp Deep Learning End-to-End
Bên cạnh phương pháp hybrid, một mô hình học sâu end-to-end cũng được triển khai dựa trên kiến trúc VGG16. Trong phương pháp này, mô hình được huấn luyện trực tiếp từ ảnh đầu vào đến nhãn đầu ra.
Dữ liệu được nạp bằng API `image_dataset_from_directory` của TensorFlow, cho phép tự động đọc ảnh và gán nhãn dựa trên cấu trúc thư mục:
```python
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_for_loader,
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=CONFIG["batch_size"],
    validation_split=0.2,
    subset='training',
    seed=42
)
```
Mô hình VGG16 được sử dụng làm backbone với trọng số pretrained từ ImageNet. Phần convolutional base được giữ nguyên (không train), và một lớp phân loại mới được thêm vào phía trên:
```python
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(*img_size, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*img_size, 3))
x = tf.keras.applications.vgg16.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
```
Mô hình được huấn luyện với optimizer Adam và hàm mất mát `sparse_categorical_crossentropy`:
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
Sau giai đoạn huấn luyện ban đầu, kỹ thuật fine-tuning được áp dụng bằng cách mở khóa một phần các lớp của mô hình VGG16 và tiếp tục huấn luyện với tốc độ học nhỏ hơn:
```python
base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
Phương pháp end-to-end cho phép mô hình học trực tiếp từ dữ liệu ảnh, tuy nhiên hiệu quả phụ thuộc nhiều vào kích thước và chất lượng của tập dữ liệu.
5. Thực nghiệm (Experiments)
5.1. Thiết lập thí nghiệm
Các thí nghiệm được thực hiện trên tập dữ liệu INRIA Person với tổng cộng 902 ảnh. Toàn bộ ảnh được resize về kích thước (224 x 224) và giữ nguyên 3 kênh màu RGB. Các tham số cấu hình chính được thiết lập như sau:
```python
CONFIG = {
    "image_size": (224, 224),
    "batch_size": 32,
    "pretrained_model": "resnet50",
    "classifier": "svm"
}
```
Đối với phương pháp Hybrid, dữ liệu sau khi trích xuất đặc trưng được chia thành tập huấn luyện và kiểm tra theo tỷ lệ 80/20:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_raw, test_size=0.2, random_state=42
)
```
Đối với phương pháp Deep Learning, dữ liệu được chia trực tiếp thông qua API của TensorFlow với tham số `validation_split=0.2`.
5.2. Kết quả phương pháp Hybrid (ResNet50 + SVM)
Sau khi trích xuất đặc trưng bằng ResNet50, mỗi ảnh được biểu diễn bởi vector 2048 chiều. Mô hình SVM được huấn luyện trên tập train và đánh giá trên tập test.
```python
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))
```
```
              precision    recall  f1-score   support

        Test       0.70      0.52      0.59        58
       Train       0.80      0.89      0.84       123

    accuracy                           0.77       181
   macro avg       0.75      0.71      0.72       181
weighted avg       0.77      0.77      0.76       181
```
Kết quả cho thấy mô hình đạt độ chính xác khoảng 77.3% trên tập kiểm tra. Các chỉ số precision, recall và F1-score cho từng lớp được thể hiện trong báo cáo phân loại.
Ngoài ra, ma trận nhầm lẫn (confusion matrix) được sử dụng để trực quan hóa hiệu suất của mô hình:
```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```
![alt text](image3.png)
Kết quả cho thấy mô hình có khả năng phân biệt tương đối tốt giữa hai lớp, tuy nhiên vẫn tồn tại một số lượng đáng kể các dự đoán sai, đặc biệt ở lớp có số lượng mẫu ít hơn.
5.3. Kết quả phương pháp Deep Learning (VGG16)
5.3.1. Transfer Learning
Mô hình VGG16 được huấn luyện với phần convolutional base được giữ nguyên và chỉ huấn luyện lớp phân loại phía trên. Sau 5 epoch, mô hình đạt độ chính xác trên tập validation khoảng 68.3% – 71.6%.
```python
history = model.fit(train_ds, validation_data=val_ds, epochs=5)
```
```
Training Deep Learning Head...
Epoch 1/5
23/23 ━━━━━━━━━━━━━━━━━━━━ 533s 23s/step - accuracy: 0.6662 - loss: 1.6694 - val_accuracy: 0.7000 - val_loss: 1.5096
Epoch 2/5
23/23 ━━━━━━━━━━━━━━━━━━━━ 518s 23s/step - accuracy: 0.6994 - loss: 1.1475 - val_accuracy: 0.7167 - val_loss: 1.1559
Epoch 3/5
23/23 ━━━━━━━━━━━━━━━━━━━━ 536s 24s/step - accuracy: 0.7202 - loss: 0.8556 - val_accuracy: 0.7111 - val_loss: 1.0923
Epoch 4/5
23/23 ━━━━━━━━━━━━━━━━━━━━ 534s 23s/step - accuracy: 0.7521 - loss: 0.6928 - val_accuracy: 0.7111 - val_loss: 1.1047
Epoch 5/5
23/23 ━━━━━━━━━━━━━━━━━━━━ 529s 23s/step - accuracy: 0.7673 - loss: 0.6027 - val_accuracy: 0.6833 - val_loss: 1.0200
Exporting features to .npy...
23/23 ━━━━━━━━━━━━━━━━━━━━ 370s 16s/step
Deep Learning process complete. Features saved: vgg16_features.npy ((722, 25088))
```
5.3.2. Fine-tuning
Sau khi huấn luyện ban đầu, một phần các lớp của VGG16 được mở khóa để fine-tune. Mô hình tiếp tục được huấn luyện thêm 5 epoch với learning rate nhỏ hơn.
```python
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)
```
Kết quả sau fine-tuning cho thấy độ chính xác đạt khoảng 71.1% trên tập validation. Mặc dù độ chính xác training tăng đáng kể, nhưng validation accuracy không cải thiện nhiều, cho thấy dấu hiệu của hiện tượng overfitting.
```
Number of layers in the base model: 19
Retraining with fine-tuning...
Epoch 5/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 603s 26s/step - accuracy: 0.7867 - loss: 0.5651 - val_accuracy: 0.6722 - val_loss: 0.8707
Epoch 6/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 603s 26s/step - accuracy: 0.9058 - loss: 0.2518 - val_accuracy: 0.7056 - val_loss: 0.8570
Epoch 7/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 602s 26s/step - accuracy: 0.9598 - loss: 0.1312 - val_accuracy: 0.6833 - val_loss: 0.8528
Epoch 8/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 570s 25s/step - accuracy: 0.9917 - loss: 0.0750 - val_accuracy: 0.6944 - val_loss: 0.8776
Epoch 9/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 605s 27s/step - accuracy: 0.9945 - loss: 0.0513 - val_accuracy: 0.7111 - val_loss: 0.9071
Epoch 10/10
23/23 ━━━━━━━━━━━━━━━━━━━━ 603s 26s/step - accuracy: 1.0000 - loss: 0.0342 - val_accuracy: 0.7111 - val_loss: 0.9252
```
6. So sánh giữa các mô hình**
6.1 So sánh trước và sau fine-tuning
Để đánh giá ảnh hưởng của quá trình fine-tuning, độ chính xác trên tập huấn luyện và tập kiểm tra được theo dõi xuyên suốt các epoch và trực quan hóa thông qua biểu đồ. Kết quả cho thấy trong giai đoạn đầu (5 epoch đầu tiên), mô hình VGG16 với transfer learning đạt độ chính xác trên tập validation dao động trong khoảng từ 68% đến 72%. Đồng thời, độ chính xác trên tập huấn luyện tăng ổn định, cho thấy mô hình đang học được các đặc trưng cơ bản từ dữ liệu.
![alt text](image4.png)
Sau khi tiến hành fine-tuning bằng cách mở khóa một phần các lớp của mạng VGG16 và tiếp tục huấn luyện với learning rate nhỏ hơn, độ chính xác trên tập huấn luyện tăng mạnh và nhanh chóng đạt gần 100%. Tuy nhiên, độ chính xác trên tập validation không có sự cải thiện tương ứng mà chỉ dao động quanh mức khoảng 68% đến 71%. Khoảng cách giữa độ chính xác của tập huấn luyện và tập validation ngày càng lớn theo số epoch, cho thấy mô hình bắt đầu ghi nhớ dữ liệu huấn luyện thay vì học được các đặc trưng có khả năng tổng quát hóa.
Từ kết quả này có thể kết luận rằng quá trình fine-tuning trong trường hợp này không mang lại cải thiện đáng kể về hiệu năng trên dữ liệu chưa thấy, mà ngược lại còn làm tăng nguy cơ overfitting. Nguyên nhân chính là do kích thước của tập dữ liệu tương đối nhỏ, không đủ để hỗ trợ việc điều chỉnh sâu các tham số của mạng nơ-ron lớn như VGG16.
6.2. So sánh tổng thể các phương pháp
Kết quả thực nghiệm của các phương pháp được tổng hợp và so sánh dựa trên độ chính xác như sau. Mô hình kết hợp ResNet50 và SVM đạt độ chính xác cao nhất, khoảng 77.3% trên tập kiểm tra. Trong khi đó, mô hình VGG16 sử dụng transfer learning đạt khoảng 68% và sau khi fine-tuning tăng lên khoảng 71.1%.
Phương pháp	Accuracy
ResNet50 + SVM	~77.3%
VGG16 (Transfer Learning)	~68%
VGG16 (Fine-tuned)	~71%
Biểu đồ so sánh cho thấy phương pháp Hybrid vượt trội hơn so với hai phương pháp Deep Learning end-to-end trong bối cảnh bài toán hiện tại. Điều này có thể được giải thích bởi đặc điểm của tập dữ liệu INRIA Person có kích thước hạn chế. Khi sử dụng ResNet50 làm bộ trích xuất đặc trưng, mô hình đã tận dụng được các đặc trưng mạnh mẽ được học từ tập dữ liệu lớn như ImageNet. Sau đó, SVM đóng vai trò là bộ phân loại hiệu quả trong không gian đặc trưng có chiều cao, giúp đạt được kết quả tốt mà không cần huấn luyện toàn bộ mạng sâu.
Ngược lại, mô hình VGG16 khi được huấn luyện theo hướng end-to-end phụ thuộc nhiều vào dữ liệu hiện có. Trong điều kiện dữ liệu nhỏ, mô hình dễ rơi vào tình trạng overfitting, đặc biệt khi tiến hành fine-tuning. Mặc dù fine-tuning giúp cải thiện độ chính xác trên tập huấn luyện và có tăng nhẹ trên tập validation, nhưng mức cải thiện này không đáng kể và không đủ để vượt qua phương pháp Hybrid.
![alt text](image2.png)
Từ toàn bộ kết quả thực nghiệm, có thể nhận thấy rằng việc kết hợp giữa mô hình học sâu để trích xuất đặc trưng và các thuật toán học máy truyền thống để phân loại là một hướng tiếp cận hiệu quả trong các bài toán có dữ liệu hạn chế. Trong khi đó, các mô hình Deep Learning end-to-end chỉ phát huy tối đa hiệu quả khi có đủ dữ liệu để huấn luyện và fine-tune một cách toàn diện.
7. Tài nguyên tham khảo và mã nguồn
Toàn bộ mã nguồn, notebook thực nghiệm và các file liên quan của bài toán được lưu trữ trên GitHub nhằm phục vụ cho việc tái hiện kết quả và tham khảo chi tiết quá trình triển khai.
GitHub Repository:
https://github.com/ngtan369/Hybrid-Image-Classification
Google Colab Notebook:
https://colab.research.google.com/github/ngtan369/Hybrid-Image-Classification/blob/scratch/ex3_imageData.ipynb
Repository bao gồm các thành phần chính như:
Module tiền xử lý dữ liệu
Pipeline trích xuất đặc trưng bằng ResNet50
Huấn luyện mô hình SVM
Huấn luyện và fine-tuning mô hình VGG16
Các notebook thực nghiệm và trực quan hóa kết quả
Google Colab được sử dụng để thực hiện huấn luyện mô hình và chạy thực nghiệm trên môi trường GPU, giúp giảm thời gian xử lý và thuận tiện cho việc tái lập kết quả.