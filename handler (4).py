import runpod
import torch
from ultralytics import YOLOE
import numpy as np
from PIL import Image
import io
import base64

# 1. CARGA DEL MODELO (Fuera del handler para evitar recargar en cada petición)
print("Iniciando carga del modelo YOLOE-26x-seg en GPU...")
try:
    # Forzamos el uso de GPU si está disponible
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLOE("yoloe-26x-seg.pt").to(device)
    print(f"✅ Modelo listo en dispositivo: {device}")
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")

def handler(job):
    """
    Función que RunPod ejecuta automáticamente al recibir un JSON
    """
    try:
        # 2. Extraer datos del JSON de entrada
        job_input = job.get("input", {})
        image_b64 = job_input.get("file")
        text_prompt = job_input.get("text_prompt", "objeto")

        if not image_b64:
            return {"error": "No se proporcionó el campo 'file' en base64"}

        # 3. Decodificar imagen Base64 a PIL -> Numpy
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        # 4. Configurar YOLOE (Dynamic Vocabulary)
        classes = [c.strip() for c in text_prompt.split(",")]
        model.set_classes(classes)

        # 5. Inferencia
        results = model.predict(img_array, verbose=False)
        
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

        return {"detections": detections}

    except Exception as e:
        return {"error": f"Error durante el procesamiento: {str(e)}"}

# 7. INICIAR EL WORKER
runpod.serverless.start({"handler": handler})
