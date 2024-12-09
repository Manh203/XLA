import cv2
import pandas as pd
import numpy as np

# Đường dẫn tới các mô hình
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

# Đọc dữ liệu từ file CSV
file_path = "age_gender.csv"  # Đường dẫn tới file CSV
data = pd.read_csv(file_path)

# Kích thước ảnh (giả định từ dữ liệu pixels)
IMAGE_SIZE = 48  # Chiều rộng và chiều cao ảnh (48x48)

# Biến lưu trữ kết quả
true_predictions = 0
false_predictions = 0
total_faces_detected = 0
correct_images = []  # Lưu ảnh đã đoán đúng

# Xử lý từng dòng trong CSV (giới hạn số lượng để kiểm tra)
max_images = 10  # Xử lý tối đa 10 ảnh để kiểm tra
for index, row in data.iterrows():
    if index >= max_images:
        break

    true_age = int(row['age'])
    true_gender = gender_classifications[int(row['gender'])]
    pixels = row['pixels']

    # Chuyển `age` thành khoảng tuổi phù hợp
    for age_range in age_classifications:
        lower, upper = map(int, age_range.strip("()").split("-"))
        if lower <= true_age <= upper:
            true_age_class = age_range
            break

    # Tái tạo ảnh từ dữ liệu pixel
    pixel_values = np.fromstring(pixels, sep=' ', dtype=np.uint8)
    if pixel_values.size != IMAGE_SIZE * IMAGE_SIZE:
        print(f"Lỗi: Kích thước pixel không khớp cho dòng {index}")
        continue

    image = pixel_values.reshape((IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Chuyển sang BGR nếu cần
    image = cv2.resize(image, (300, 300))  # Resize ảnh để phù hợp với mô hình
    img_cp = image.copy()
    img_h, img_w = img_cp.shape[:2]
    blob = cv2.dnn.blobFromImage(img_cp, 0.3, (300, 300), MODEL_MEAN_VALUES, True, False)

    face.setInput(blob)

    try:
        # Thêm timeout cho `face.forward`
        detected_faces = face.forward()
    except Exception as e:
        print(f"Lỗi khi phát hiện khuôn mặt dòng {index}: {e}")
        continue

    for i in range(detected_faces.shape[2]):
        confidence = detected_faces[0, 0, i, 2]
        if confidence > 0.99:
            x1 = int(detected_faces[0, 0, i, 3] * img_w)
            y1 = int(detected_faces[0, 0, i, 4] * img_h)
            x2 = int(detected_faces[0, 0, i, 5] * img_w)
            y2 = int(detected_faces[0, 0, i, 6] * img_h)
            total_faces_detected += 1

            # Điều chỉnh tọa độ sau khi resize ảnh
            y1_pad, y2_pad = max(0, int(y1 * 300 / img_h) - 15), min(int(y2 * 300 / img_h) + 15, 300)
            x1_pad, x2_pad = max(0, int(x1 * 300 / img_w) - 15), min(int(x2 * 300 / img_w) + 15, 300)

            # Kiểm tra xem tọa độ có hợp lệ không
            if x1_pad < x2_pad and y1_pad < y2_pad:
                face_img = img_cp[y1_pad:y2_pad, x1_pad:x2_pad]

                # Kiểm tra vùng cắt không rỗng
                if face_img.size != 0:
                    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, True)

                    # Dự đoán giới tính
                    gen.setInput(blob)
                    gender_prediction = gen.forward()
                    predicted_gender = gender_classifications[gender_prediction[0].argmax()]

                    # Dự đoán độ tuổi
                    age.setInput(blob)
                    age_prediction = age.forward()
                    predicted_age_class = age_classifications[age_prediction[0].argmax()]

                    # So sánh với nhãn thực tế
                    if predicted_gender == true_gender and predicted_age_class == true_age_class:
                        true_predictions += 1
                        # Vẽ hình chữ nhật và thêm chú thích cho ảnh đúng
                        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_cp, f'{predicted_gender}, {predicted_age_class}',
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        correct_images.append(img_cp)  # Lưu ảnh đã đoán đúng
                    else:
                        false_predictions += 1
                else:
                    print(f"Lỗi: Vùng ảnh rỗng tại dòng {index}")
            else:
                print(f"Lỗi: Tọa độ vùng cắt không hợp lệ tại dòng {index}")

# Tính độ đo
if total_faces_detected > 0:
    accuracy = (true_predictions / total_faces_detected) * 100
    error_rate = 100 - accuracy
    print(f"Tổng số khuôn mặt phát hiện: {total_faces_detected}")
    print(f"Dự đoán đúng: {true_predictions}")
    print(f"Dự đoán sai: {false_predictions}")
    print(f"Độ chính xác (Accuracy): {accuracy:.2f}%")
    print(f"Tỷ lệ lỗi (Error Rate): {error_rate:.2f}%")
else:
    print("Không phát hiện được khuôn mặt nào.")

# Hiển thị ảnh đã đoán đúng
if correct_images:
    for i, img in enumerate(correct_images):
        cv2.imshow(f"Correct Prediction {i+1}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Không có ảnh nào đoán đúng.")
