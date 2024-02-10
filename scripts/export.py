from ultralytics import YOLO

model = YOLO('path/to/best.pt')
model.export(format='engine', half=True)

# export process will run, any missing libraries will attempt to install

# Then to load for inference
# The task argument must be set
# model = YOLO('path/to/best.pt', task='segment')
# model = YOLO('path/to/best.pt', task='detect')
model = YOLO('path/to/export.engine',task='segment')
results = model.predict(source='your/source/here', stream=False, device='cuda:0')