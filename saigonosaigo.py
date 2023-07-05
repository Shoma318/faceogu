import streamlit as st
from streamlit_webrtc import webrtc_streamer,WebRtcMode,VideoProcessorBase
from streamlit_toggle import toggle
import cv2
import numpy as np
import av
from deepface import DeepFace
import time
import queue
import logging
from typing import List, NamedTuple
# タイマー開始
tic = time.time() - 5
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', level=logging.DEBUG)
mode = st.sidebar.radio("mode",('original', 'gray', 'edge'),horizontal=True)
thmin = st.sidebar.slider("Min",min_value=0,max_value=100,value=32)
thmax = st.sidebar.slider("Max",min_value=0,max_value=200,value=55)
# 顔解析を何秒ごとに実行するか。この数値で指定した時間スルーする
time_threshold = st.sidebar.slider("time threshold",min_value=1,max_value=10,value=5)
# 顔が何フレームつづけて検出されたら顔とみなすか
frame_threshold = st.sidebar.slider("frame threshold",min_value=1,max_value=10,value=5)
    
face_queue = queue.Queue(maxsize=1)
st.title('Streamlit')
face_included_frames = 0

def videoProcessor(frame: av.VideoFrame) -> av.VideoFrame:
    global tic,face_included_frames
    cimg = frame.to_ndarray(format = 'bgr24') # カラー画像
    cimg = cv2.flip(cimg,1)
    cimg = cv2.resize(cimg,(640,480))
    if mode != 'original':
        gimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY) # グレイ画像
        if mode == 'edge':
            gimg = cv2.Canny(gimg, thmin, thmax)# エッジ画像
        eimg = cv2.cvtColor(gimg, cv2.COLOR_GRAY2BGR) 
    else:
        eimg = cimg.copy()
        
    toc = time.time()
    if (toc-tic) > time_threshold: # time_threshold 時間経過するまではスルー
        face_detected = False
        try: # 顔認識を試みる
            faces = DeepFace.extract_faces(img_path=cimg,target_size = (224, 224),detector_backend = 'opencv')
        except Exception as e:
            # logging.error("顔認識でエラーがでました",e)
            faces=[]
        if len(faces)==0: # 顔が検出されなかった場合
            cv2.rectangle(eimg, (0, 0), (639, 479), (167,167,210), 20) 
        else: # 顔が検出された場合
            face_detected = False # 有効なサイズの顔が見つかったかどうかのフラグ
            for face in faces:
                rct = face['facial_area']
                x,y,w,h = rct['x'],rct['y'],rct['w'],rct['h']
                if w > 130: # 幅130以上でないと顔検出とみなさない
                    fflag = True
                    cv2.rectangle(eimg, (0, 0), (639, 479), (128,200,128), 20) 
                    face_detected = True
                    # faceimg = eimg[y:y+h,x:x+w]
                    keepface = (cimg,(x,y,w,h))
                    cv2.rectangle(eimg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    break
            if face_detected: # 顔フレームカウンタの表示
                face_included_frames = face_included_frames + 1
                cv2.putText(eimg, str(frame_threshold - face_included_frames),
                            (int(x + w / 4), int(y + h / 1.5)),
                            cv2.FONT_HERSHEY_SIMPLEX,4,
                            (255, 255, 255),2)
        if face_included_frames >= frame_threshold:                     
            if face_queue.full():
                face_queue.get()
            # face_queue.put(faceimg)
            face_queue.put(keepface)
            tic = time.time()
            face_included_frames = 0
    else: # 認識開始までのカウントダウンタイマーの表示
        cv2.putText(eimg, str(round(time_threshold-(toc-tic))),
                    (0,100), cv2.FONT_HERSHEY_SIMPLEX,4,(255, 255, 255),2)
    frame = av.VideoFrame.from_ndarray(eimg, format='bgr24')    
    return frame
webrtc = webrtc_streamer(
        key="abc", 
        mode = WebRtcMode.SENDRECV,
        # video_processor_factory=VideoProcessor,
        video_frame_callback=videoProcessor,
        rtc_configuration={ "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True,"audio": False},
        async_processing=True,)

import pandas as pd
if webrtc.state.playing:
    col1, col2, col3 = st.columns(3)
    faceimage = col1.empty()
    analysis = col2.empty()
    emotions = col3.empty()
    
    while True:
        (img,(x,y,w,h)) = face_queue.get()
        face = img[y:y+h,x:x+w]
        objs = DeepFace.analyze(img_path = face, 
        actions = ['age', 'gender', 'race', 'emotion'],enforce_detection=False)[0]
        if mode == 'edge' or mode == 'gray':
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            if mode == 'edge':
                face = cv2.Canny(face,thmin,thmax)
            face = cv2.cvtColor(face,cv2.COLOR_GRAY2RGB)
        else:
            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)        
        faceimage.image(face)      
        df1 = pd.DataFrame(index=['性別','年齢','人種','表情'],columns =['値'],
                           data=[objs['dominant_gender'],str(objs['age']),
                            objs['dominant_race'],objs['dominant_emotion']])
        df2 = pd.DataFrame((objs['emotion']).items(), columns=["emotion", "score"])
        df2 = df2.sort_values(by=["score"], ascending=False).reset_index(drop=True)
        analysis.table(df1)
        emotions.table(df2)