
# Team: [HCMUS - FIT]
# Competition: Aero-Eyes Challenge


## Folder structure

Cấu trúc thư mục được tổ chức như sau:


```
submission_folder/
│
├── Dockerfile
│
├──  code/
│   ├── analyzing-data.ipynb
│   ├── Preprocess.ipynb                       # visdrone dataset 
│   ├── Inference.ipynb             
│   ├── Finetune.ipynb
│   └── predict_notebook.ipynb                 # Notebook đo lường thời gian 
│   
│
├── weights/                        # Thư mục chứa weight model đã fine-tune
│   └── best.pt                     # File weight model
│
├── data/
│   └── test/
│       └── samples/
│           └── [từng thư mục test_case, ví dụ: drone_video_001/]
│
└── README.md
```

---

## Data

**Cấu trúc thư mục test (theo BTC yêu cầu):**
```
│
├── data/
│   └── test/
│       └── samples/
│           └── [từng thư mục test_case, ví dụ: drone_video_001/]
```

- Trong `samples/`, mỗi thư mục con là **1 test case**  
- Mỗi test case chứa:  
  - `images/`: 3 ảnh tham chiếu (ví dụ frames, bounding boxes, hoặc metadata)  
  - `dronevideo.mp4`: video chính để mô hình dự đoán


---


---

## Idea

### Preprocess.ipynb
- **Tiền xử lý dữ liệu** trước khi huấn luyện
- **Hợp nhất** (Merge) **nhiều bộ dữ liệu** (e.g., VisDrone) vào một cấu trúc thống nhất.
- **Chuyển đổi** và sắp xếp dữ liệu ảnh/label sang **định dạng chuẩn của YOLO** (tách riêng `images/` và `labels/`).
- **Thao tác chính:** Đọc đường dẫn, sao chép/di chuyển file, kiểm tra tính toàn vẹn của dữ liệu.
- **Output:** Thư mục dataset đã chuẩn hóa, bao gồm `images/train`, `labels/train`, `images/val`, `labels/val` và file cấu hình `dataset.yaml`.

### Finetune.ipynb
- **Huấn luyện / Fine-tune** mô hình **YOLO11s** trên tập dữ liệu đã xử lý.
- Load base model từ một checkpoint đã có (ví dụ: `/path/to/yolo11s/train/weights/best.pt` hoặc từ một URL).
- **Set seed cố định** để đảm bảo tính tái tạo (reproducibility) của quá trình huấn luyện.
- Áp dụng các **kỹ thuật tăng cường dữ liệu** (Augmentation) phức tạp trong quá trình training.
- Save checkpoint cuối cùng (`best.pt`) vào thư mục `runs/`.

### Inference.ipynb
- **Load model** từ trọng số đã tinh chỉnh (`best.pt`).
- **Read test cases** từ thư mục test (có cấu trúc video và ảnh kèm theo).
- Thực hiện **Object Tracking (theo dõi vật thể)** trên video test case (`model.track()`).
- **Run inference** cho từng test case và đo thời gian thực thi.
- **Output:**
  - `jupyter_time_submission.csv`
  - `jupyter_submission.json`

---

## Training & Inference

### Training
- **Seed cố định:** **`2024`** 
- **Optimizer:** **`SGD`** (Stochastic Gradient Descent)
- **Learning rate (lr0):** **`0.0001`**
- **Epochs:** **`80`**
- **Batch size:** **`16`**
- **Data augmentations:** **`hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `translate=0.1`, `scale=0.5`, `fliplr=0.5`** và các kỹ thuật khác như **Mosaic/Mixup**.
- **Checkpoints:** Lưu `best.pt` (trọng số tốt nhất) vào thư mục `outputs/` của quá trình chạy.

### Inference
- **Load** mô hình **YOLO11s** từ `best.pt`.
- **Pipeline:** Load input (video/ảnh) $\rightarrow$ `model.track()` $\rightarrow$ Postprocess (NMS, trích xuất ID, Box) $\rightarrow$ Format JSON.
- **Output:** Ghi kết quả dự đoán (bao gồm cả ID đối tượng và tọa độ) vào `jupyter_submission.json` và thời gian chạy vào `jupyter_time_submission.csv`.
- **Notebook đo thời gian theo cell riêng:**
  1. Set seed
  2. Load model
  3. Inference loop (Bắt đầu và kết thúc đo thời gian $t_1, t_2$ cho từng test case)

---

## 6. Run docker 

``` bash 
  docker run -it --rm --gpus all -p 8888:8888 -v ${PWD}/weight:/src/weight -v ${PWD}/data/test:/src/data/test -v ${PWD}/result:/result --name zaloai zaloai-python39 jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
```

## 7. Requirements

Danh sách các thư viện cần thiết cho quá trình Inference (và Training) phải được cài đặt trong môi trường Docker:

- **Python >= 3.9**
- **torch** 
- **ultralytics** (Phiên bản >= 8.3.x)
- **numpy**
- **Pillow (PIL)** 
- **tqdm**
- **opencv-python** 
