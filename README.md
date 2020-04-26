# wcp
Watch Car Plate

Objective: Build the Taiwan car plate recognition mechanism

Scope:
1. AI modeling for Taiwan car plate recognition
2. Front-end integration of Nvidia Jetson/ APP/ Line Chatbot
3. Back-end API service development

AI model solution:
1. Adopt YoloV3 to detect car plate object, and then pass detected object to CRNN for text recognition.
2. Train the CRNN text recognition model via in-house pseudo car plate photo generator initially; further fit the model via real car plate photos collected.
3. Train the YoloV3 object detection model via real car plate photos collected.
4. YoloV3 refer to https://github.com/qqwweee/keras-yolo3
5. CRNN refer to https://github.com/qjadud1994/CRNN-Keras

Directories and progress:
1. *plategen* ==> pseudo car plate generator; progress 50%, ready for new SPEC car plate of Taiwan
2. *crnn* ==> CRNN model; progress 50%, pseudo car plate training OK, but need advanced training for real car plate; collecting real photos
3. *yolov3* ==> YoloV3 model; progress 0%, proved the concept to roughly bound the car plate
