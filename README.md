# KNN-Internship-Division-Recommendation
This project implements a machine learning model using the K-Nearest Neighbor (KNN) algorithm to recommend the most suitable internship division for students based on their profile.  The system was developed as part of an undergraduate thesis titled:

**"Pengembangan Sistem Informasi Administrasi Magang Berbasis Web dengan Penerapan Algoritma K-Nearest Neighbor (KNN) untuk Rekomendasi Divisi Magang."**

---

# Project Overview

The goal of this project is to assist the internship administration process by providing **automatic recommendations of internship divisions** based on historical internship data.

The model analyzes several student attributes such as:

* Academic background
* Skills
* Interests
* Previous internship patterns

and predicts the most suitable division using the **K-Nearest Neighbor classification algorithm**.

---

# Dataset

The dataset used in this project contains internship participant records including:

| Feature | Description                |
| ------- | -------------------------- |
| Jurusan | Student major              |
| Skill   | Technical skill            |
| Minat   | Student interest           |
| IPK     | GPA                        |
| Divisi  | Target internship division |

File:

```
dataset/dataset_magang.csv
```

---

# Machine Learning Pipeline

1. Data preprocessing
2. Feature encoding
3. Model training using KNN
4. Model serialization using Pickle
5. Prediction via API

---

# Result

The implemented KNN model successfully generates internship division recommendations based on student attributes such as major, skills, and interests. The model was trained using historical internship data and deployed using a lightweight API for prediction. The system demonstrates how machine learning can assist administrative decision-making in internship placement.

---

# Model Files

The trained model and preprocessing objects are stored as serialized files:

```
model/model_knn.pkl
model/preprocessor.pkl
model/label_encoder.pkl
model/le_divisi.pkl
```

These files allow the model to be reused without retraining.

---

# Project Structure

```
dataset/      -> dataset used for training
model/        -> trained machine learning model
src/          -> source code for model and API
config/       -> feature metadata and configuration
```

---

# Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Pickle
* JSON

---

# API Usage

Prediction can be accessed through the API script:

```
python src/api_knn.py
```

The API will load the trained model and return internship division recommendations based on input features.

---

# Author

Silvy Putri Hanafi
Informatics Engineering
Undergraduate Thesis Project

---

# License

This project is developed for academic and research purposes.
