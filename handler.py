import runpod
import torch
import requests # Necesario para descargar JPG desde URL
from ultralytics import YOLOWorld
import numpy as np
from PIL import Image
import io
import base64

# 1. CARGA DEL MODELO
print("Iniciando carga del modelo yolov8x-worldv2.pt en GPU...")
try:
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLOWorld("yolov8x-worldv2.pt").to(device)
    print(f"✅ Modelo listo en dispositivo: {device}")
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")

def handler(job):
    try:
        # 2. Extraer datos del JSON de entrada
        job_input = job.get("input", {})
        image_source = job_input.get("file")
        text_prompt = job_input.get("text_prompt", "objeto")

        # Parámetros de inferencia con valores por defecto
        conf_thresh = float(job_input.get("conf", 0.25))
        iou_thresh = float(job_input.get("iou", 0.45))
        img_size = int(job_input.get("imgsz", 640))
        max_detections = int(job_input.get("max_det", 300))

        if not image_source:
            return {"error": "No se proporcionó el campo 'file' (Base64 o URL)"}

        # --- 3. LÓGICA DE CARGA DE IMAGEN (JPG URL o Base64) ---
        if image_source.startswith("http"):
            # Es una URL JPG
            response = requests.get(image_source, timeout=10)
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            # Es Base64
            if "," in image_source:
                image_source = image_source.split(",")[1]
            image_bytes = base64.b64decode(image_source)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        img_array = np.array(img)

        # 4. Configurar Vocabulario Dinámico
        classes = [c.strip() for c in text_prompt.split(",")]
        model.set_classes(classes)

        # 5. Inferencia
        results = model.predict(
            img_array, 
            conf=conf_thresh, 
            iou=iou_thresh, 
            imgsz=img_size, 
            max_det=max_detections,
            verbose=False
        )
        
        # 6. Formatear respuesta JSON
        detections = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    detections.append({
                        "class": classes[cls_ids[i]] if cls_ids[i] < len(classes) else "unknown",
                        "confidence": round(float(confs[i]), 4),
                        "bbox": [round(float(x), 2) for x in boxes[i].tolist()]
                    })

        return {
            "detections": detections,
            "params_used": {
                "imgsz": img_size,
                "conf": conf_thresh,
                "source_type": "url" if image_source.startswith("http") else "base64"
            }
        }

    except Exception as e:
        return {"error": f"Error durante el procesamiento: {str(e)}"}

# 7. INICIAR EL WORKER
runpod.serverless.start({"handler": handler})
