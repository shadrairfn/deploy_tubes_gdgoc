import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="Deteksi Sampah dengan YOLO",
    page_icon="♻️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load YOLOv8 model"""
    model_path = "D:\\ML & DL\\GARBAGE 2.0\\fooldeer_v8\\runs\\detect\\train2\\weights\\best.pt"  
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di {model_path}")
        return None
    return YOLO(model_path)

def main():
    st.title("Deteksi Sampah Organik dan Anorganik")
    
    st.sidebar.title("Pengaturan")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    classNames = ['Anorganik', 'Organik']
  
    with st.spinner("Loading model..."):
        model = load_model()
        if model is None:
            st.warning("Silakan sesuaikan path model pada script")
            return
    
    uploaded_file = st.file_uploader("Unggah gambar sampah", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
        st.image(img_rgb, caption="Gambar yang diunggah", use_column_width=True)
        
        if st.button("Deteksi Sampah"):
            with st.spinner("Melakukan deteksi..."):
                results = model(img)
                
                annotated_img = img_rgb.copy()
                boxes = results[0].boxes
                
                organik_count = 0
                anorganik_count = 0
                detected_items = []
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf > confidence:
                        label = f"{classNames[cls]}: {conf:.2f}"
                        
                        if classNames[cls] == "Organik":
                            organik_count += 1
                        else:
                            anorganik_count += 1
                        
                        detected_items.append({
                            "type": classNames[cls],
                            "confidence": conf,
                            "position": (x1, y1, x2, y2)
                        })
                        
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        cv2.rectangle(annotated_img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (255, 255, 255), -1)
                        cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                st.image(annotated_img, caption="Hasil Deteksi", use_column_width=True)
                
                if len(detected_items) > 0:
                    st.write("### Hasil Deteksi")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Sampah Organik", organik_count)
                    col2.metric("Sampah Anorganik", anorganik_count)
                    
                    st.write("### Detail Objek Terdeteksi")
                    detection_data = []
                    for i, item in enumerate(detected_items):
                        detection_data.append({
                            "No": i+1,
                            "Jenis": item["type"],
                            "Confidence": f"{item['confidence']:.2f}"
                        })
                    
                    st.table(detection_data)
                else:
                    st.info("Tidak ada objek yang terdeteksi dengan confidence threshold yang dipilih")
if __name__ == "__main__":
    main()