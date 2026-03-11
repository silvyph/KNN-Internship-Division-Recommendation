from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Muat model dan preprocessing
try:
    knn = joblib.load('model_knn.pkl')
    le_divisi = joblib.load('label_encoder.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    print("✅ Model dan preprocessing loaded successfully!")
    print(f"   Model classes: {le_divisi.classes_}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Pastikan file berikut ada dan valid:")
    print("  - model_knn.pkl")
    print("  - label_encoder.pkl") 
    print("  - preprocessor.pkl")
    exit()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": True,
        "n_classes": len(le_divisi.classes_)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"🔧 Data received: {data}")

        # Validasi input wajib
        required_fields = ['jurusan', 'mapel1', 'mapel2', 'skill_teknis', 'sertifikasi', 'proyek', 'tanggal_mulai', 'tanggal_akhir']
        
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({
                    "error": f"Field '{field}' wajib diisi"
                }), 400

        # Ekstrak dan proses data
        jurusan = data['jurusan'].strip().lower()
        mapel1 = data['mapel1'].strip().lower()
        mapel2 = data['mapel2'].strip().lower()
        skill_teknis = data['skill_teknis'].strip().lower()
        sertifikasi = data['sertifikasi'].strip().lower()
        proyek = data['proyek'].strip().lower()
        
        # Hitung durasi dari tanggal
        try:
            start_date = datetime.strptime(data['tanggal_mulai'], '%Y-%m-%d')
            end_date = datetime.strptime(data['tanggal_akhir'], '%Y-%m-%d')
            durasi_hari = (end_date - start_date).days
            
            if durasi_hari <= 0:
                return jsonify({
                    "error": "Tanggal akhir harus setelah tanggal mulai"
                }), 400
                
        except ValueError as e:
            return jsonify({
                "error": "Format tanggal harus YYYY-MM-DD"
            }), 400

        # Buat DataFrame input
        input_df = pd.DataFrame({
            'jurusan': [jurusan],
            'mapel1': [mapel1],
            'mapel2': [mapel2],
            'skill_teknis': [skill_teknis],
            'sertifikasi': [sertifikasi],
            'proyek': [proyek],
            'durasi_hari': [durasi_hari]
        })

        print(f"📊 Input DataFrame:\n{input_df}")

        # Preprocessing
        try:
            X_processed = preprocessor.transform(input_df)
            print(f"✅ Preprocessing successful. Shape: {X_processed.shape}")
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return jsonify({
                "error": "Error dalam preprocessing data",
                "details": str(e)
            }), 400

        # Prediksi
        try:
            y_pred = knn.predict(X_processed)[0]
            divisi_pred = le_divisi.inverse_transform([y_pred])[0]
            
            # Dapatkan confidence score
            proba = knn.predict_proba(X_processed)[0]
            confidence = float(proba[y_pred])
            
            print(f"🎯 Prediction: {divisi_pred} (confidence: {confidence:.3f})")

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return jsonify({
                "error": "Error dalam melakukan prediksi",
                "details": str(e)
            }), 500

        # Response
        return jsonify({
            "success": True,
            "predicted_divisi": divisi_pred,
            "confidence": round(confidence, 3),
            "input_data": {
                "jurusan": jurusan,
                "mapel1": mapel1,
                "mapel2": mapel2,
                "skill_teknis": skill_teknis,
                "sertifikasi": sertifikasi,
                "proyek": proyek,
                "durasi_hari": durasi_hari
            }
        })

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return jsonify({
            "error": "Terjadi kesalahan internal server",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server for KNN model...")
    print("📡 API akan berjalan di http://127.0.0.1:5000")
    print("📋 Endpoint: POST /predict")
    print("💡 Pastikan model sudah di-training sebelum menggunakan API ini")
    
    app.run(host='0.0.0.0', port=5000, debug=True)