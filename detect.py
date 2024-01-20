from ultralytics import YOLO

model = YOLO("best.onnx")

model.predict(source="pure1.mp4", show=True, save=True, show_labels=True, show_conf=True, conf=0.5, save_txt=False, save_conf=False, line_width=2, box=False)

model.export(format="onnx")