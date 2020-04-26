# wcp
Watch Car Plate

目標: 建立任意角度台灣車牌辨識機制

範圍:
1. AI 模型 for 車牌辨識
2. 前端 Nvidia Jetson/ APP/ Line Chatbot 整合
3. 後端 API Service 建置

AI 模型解決方案:
1. 以 YoloV3 負責車牌物件偵測，將偵測出的車牌影像擷取後傳送至 CRNN model 進行車牌文字辨識
2. YoloV3 refer to https://github.com/qqwweee/keras-yolo3
3. CRNN refer to https://github.com/qjadud1994/CRNN-Keras
4. 自建車牌模擬器訓練 CRNN 文字辨識模型，訓練至一定程度再以真實車牌照片進一步訓練
5. 直接以真實照片訓練 YoloV3 物件偵測車牌模型

目錄與進度說明:
1. *plategen* ==>  car plate generator; progress 50%, ready for new SPEC car plate of Taiwan
2. *crnn* ==> CRNN model; progress 50%, pseudo car plate training OK, but need advanced training for real car plate; collecting real photos
3. *yolov3* ==> YoloV3 model; progress 0%, proved the concept to roughly bound the car plate
