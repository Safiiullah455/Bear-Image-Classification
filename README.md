# Bear-Image-Classification


# Bear Image Classification 🐻

This project is a **binary image classification system** that predicts whether an image contains a **bear or not a bear** using a **CNN-based deep learning model**.

The trained model is served using **FastAPI** for easy testing via API.

---

## 📂 Project Structure

Bear-Image-Classification/
│── Api.py                # FastAPI app for model testing
│── BearCnn.py            # Model training script
│── BearCnn.h5            # Trained CNN model
│── Beaardata.csv         # Dataset labels (paths + classes)

## 📊 Dataset

The dataset is **not included** in this repository due to size limits.

📥 **Download Dataset from:**  
👉 "https://drive.google.com/file/d/1vo7u3s88je0PE9JSRHbpUnFpJqvH1u_i/view?usp=sharing"

After downloading:
1. Extract the dataset
2. Place it anywhere on your system
3. **Update dataset paths** inside `BearCnn.py` according to your local directory

Example:
```python
DATASET_PATH = "D:/datasets/bear_images/"
````

---

## 🧠 Model Training

The model is trained using **Convolutional Neural Networks (CNN)**.

### Train the Model

Run the following command:

```bash
python BearCnn.py
```

This will:

* Load the dataset
* Train the CNN model
* Save the trained model as:

```
BearCnn.h5
```

---

# Model Testing with FastAPI

FastAPI is used to test the trained model via an API.

### Run the API Server

```bash
uvicorn Api:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

### Test Endpoint

* Upload an image
* API returns prediction: **Bear / Not Bear**

---

## ⚙️ Requirements

Install required libraries:

```bash
pip install tensorflow fastapi uvicorn numpy pandas opencv-python
```



## 👨‍💻 Author

**Safiullah**
Final Year BSCS | AI & Machine Learning
