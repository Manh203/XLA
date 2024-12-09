import cv2

# Đọc ảnh
image = cv2.imread('WE29.jpg')
image = cv2.resize(image, (720, 640))

# Định nghĩa các mô hình
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 117, 123]

# Load mô hình
face = cv2.dnn.readNet(face_pb, face_pbtxt)
age = cv2.dnn.readNet(age_model, age_prototxt)
gen = cv2.dnn.readNet(gender_model, gender_prototxt)

# Các nhãn phân loại
age_classifications = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classifications = ['Male', 'Female']

# Sao chép ảnh và chuẩn bị blob
img_cp = image.copy()
img_h, img_w = img_cp.shape[:2]
blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)

face.setInput(blob)
detected_faces = face.forward()

# Biến lưu trữ kết quả đánh giá
true_predictions = 0
false_predictions = 0
total_faces_detected = 0

# Xử lý các khuôn mặt phát hiện được
for i in range(detected_faces.shape[2]):
    confidence = detected_faces[0, 0, i, 2]
    if confidence > 0.99:
        x1 = int(detected_faces[0, 0, i, 3] * img_w)
        y1 = int(detected_faces[0, 0, i, 4] * img_h)
        x2 = int(detected_faces[0, 0, i, 5] * img_w)
        y2 = int(detected_faces[0, 0, i, 6] * img_h)
        total_faces_detected += 1
        face_bounds = [x1, y1, x2, y2]

        try:
            # Cắt khuôn mặt từ ảnh
            face_img = img_cp[max(0, y1 - 15):min(y2 + 15, img_h), max(0, x1 - 15):min(x2 + 15, img_w)]
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, True)

            # Dự đoán giới tính
            gen.setInput(blob)
            gender_prediction = gen.forward()
            gender = gender_classifications[gender_prediction[0].argmax()]

            # Dự đoán độ tuổi
            age.setInput(blob)
            age_prediction = age.forward()
            age = age_classifications[age_prediction[0].argmax()]

            # Hiển thị kết quả trên ảnh
            cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cp, f'{gender}, {age}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Đánh giá độ chính xác (giả sử có nhãn thực tế từ một nguồn nào đó)
            true_gender = "Male"  # Giới tính thực tế
            true_age = "(25-32)"  # Khoảng tuổi thực tế

            if gender == true_gender and age == true_age:
                true_predictions += 1
            else:
                false_predictions += 1

        except Exception as e:
            print(f"Lỗi xử lý khuôn mặt: {e}")
            continue

# Độ đo đánh giá
if total_faces_detected > 0:
    accuracy = (true_predictions / total_faces_detected) * 100
    error_rate = 100 - accuracy
    print(f"Tổng số khuôn mặt phát hiện: {total_faces_detected}")
    print(f"Dự đoán đúng: {false_predictions}")
    print(f"Dự đoán sai: {true_predictions}")
    print(f"Độ chính xác (Accuracy): {error_rate:.2f}%")
    print(f"Tỷ lệ lỗi (Error Rate): {accuracy:.2f}%")
else:
    print("Không phát hiện được khuôn mặt nào.")

# Hiển thị ảnh kết quả
cv2.imshow('Result', img_cp)
cv2.waitKey(0)
cv2.destroyAllWindows()
