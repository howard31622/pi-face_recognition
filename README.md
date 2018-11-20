# pi-face_recognition
this is for putting my code


＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
get data的部分

會每次拍攝20張照片作為encoding的部分

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
test data的部分

會先再拍一張作為output的測試

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
encoding的部分

主要把照片拿去做成128D，每個id都有屬於他的pickle檔

要做輸入的指令為 --encoding XXX.pickle --dataset dataset/XXX


11/16(有討論到的）

程式碼的編排

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
recognize的部分

一. 先做RFID的偵測

二. 做picamera拍照的動作

三. 進行recognize的動作

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

收取Data的方式：
兩週內在實驗室的人臉都做紀錄，並且再利用人工的方式做挑選照片

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

實驗流程：

分為收集測試資料以及作直接的辨識功能

在辨識功能中：

會先利用RFID去做為像是身份的ID，感應完後會進行拍照(而那個照片會記錄下來成為training data)，
再進行recognize，會輸出一張圖片並且會框起來人臉的部分並顯示名字

預定收集時間是2週

人數設定10人





＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
如果只要實驗辨識的話
    
    python recognize_faces_image.py
    
如果要先收集資料可以先
    
    python getData.py
    
收集完照片要做encoding

    python encode_faces.py 
    

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
關於使用到的 face_recognition package 
(網址：https://face-recognition.readthedocs.io/en/latest/face_recognition.html#face_recognition.api.compare_faces)

1.face_recognition.compare_faces
       
       face_recognition.api.compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6)

意思： Compare a list of face encodings against a candidate encoding to see if they match.

Parameters:	
   known_face_encodings – A list of known face encodings
   
   face_encoding_to_check – A single face encoding to compare against the list
   
   tolerance – How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    
Returns:	

   A list of True/False values indicating which known_face_encodings match the face encoding to check

2.face_recognition.face_distance

    face_recognition.api.face_distance(face_encodings, face_to_compare)

意思： Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. The distance tells you how similar the faces are.

Parameters:	

   faces – List of face encodings to compare
    
   face_to_compare – A face encoding to compare against
   
Returns:	
   A numpy ndarray with the distance for each face in the same order as the ‘faces’ array
   
3.face_recognition.face_encodings

    face_recognition.api.face_encodings(face_image, known_face_locations=None, num_jitters=1)
    
意思： Given an image, return the 128-dimension face encoding for each face in the image.

Parameters:	

   face_image – The image that contains one or more faces
   
   known_face_locations – Optional - the bounding boxes of each face if you already know them.
   
   num_jitters – How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)

Returns:	
   
   A list of 128-dimensional face encodings (one for each face in the image)
    
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
11/20

1.test data和收集data方式


下次要做到的部分： 把整個系統做好

11/16
下次要做到的部分：

1.可以建立各自的encoding檔（pickle）

2.拍照做data的程式要完成

3.收集兩週進實驗室的人臉

要做的預備部分：

如果以上的實驗結果失敗，會試著使用建model來測試

