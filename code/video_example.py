import cv2
import numpy as np
import tflite_runtime.Interpreter as tflite

model_path = './tflite_model.tflite'  # 모델 파일 경로 설정
interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

video_path = 'your_video.mp4'  # 비디오 파일 경로 설정
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 조정 및 전처리 (TFLite 모델에 맞게)
    input_data = cv2.resize(
        frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5  # 모델에 따라 정규화

    # 모델 실행
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 여기에서 output_data를 사용하여 필요한 작업 수행 (예: 출력을 화면에 그리기)

    # 프레임 표시
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
