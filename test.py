#here is the single picture test
import torch
import cv2
from model import VGGNetwork


def test(type='happy'):
    assert (type == 'happy' or type == 'neutral' or type == 'angry')
    emotions = ['angry', 'disgusted', 'scared', 'happy', 'sad', 'suprised', 'neutral']
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if type == 'happy':
        img = cv2.imread('test_images/happy.jpg')
    elif type == 'neutral':
        img = cv2.imread('test_images/neutral.jpg')
    elif type == 'angry':
        img = cv2.imread('test_images/angry.jpg')
    if torch.cuda.is_available():
        state_dict = torch.load('model.pth')
    else:
        state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
    network = VGGNetwork()
    network.load_state_dict(state_dict)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(img_gray, 1.3, 5)

    x_display, y_display, w_display, h_display = 0, 0, 0, 0
    biggest_area = 0

    for (x, y, w, h) in faces_detected:
        if w * h > biggest_area:
            biggest_area = w * h
            x_display, y_display, w_display, h_display = x, y, w, h
    if len(faces_detected) == 0:
        cv2.putText(img, 'FACE NOT FOUND', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cropped_face = cv2.resize(img_gray[y_display:y_display + w_display, x_display:x_display + h_display],
                                  (48, 48))  # crop face from the image and resize it to 48x48 pixels
        preprocessed_image = cropped_face / 255
        image = torch.Tensor(preprocessed_image).unsqueeze(dim=0).unsqueeze(dim=0)
        pred = network(image).argmax(dim=1)
        emoji = cv2.imread('emojis/' + emotions[pred] + '.png')
        y_image = y_display - 100
        x_image = x_display - 100
        if y_image < 0:
            y_image = 0
            y_display = 100
        if x_image < 0:
            x_image = 0
            x_display = 100
        y_text = y_image
        if y_text <= 5:
            y_text = y_display
        img[y_image:y_display, x_image:x_display] = cv2.resize(emoji, (100, 100))
        cv2.putText(img, emotions[pred], (x_display, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow('Emotion detector ', img)

    print('Press "q" to close window.')
    cv2.waitKey(0)


test('angry')
