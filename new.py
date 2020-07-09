from flask import Flask, render_template, request, redirect, Response ,url_for,jsonify
#from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, Column
from sqlalchemy.orm import sessionmaker
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
import time
import json
import cv2

from pandas.io.json import json_normalize
import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import datetime
from utils import visualization_utils as vis_util
from utils import backbone
import matplotlib.pyplot as plt


from  gevent.pywsgi import WSGIServer
from gevent import monkey
monkey.patch_all()

stop_counting = 0
file_road=""

def object_counting():
        global stop_counting
        #(0, detection_graph, category_index,0,24,720,480)
        detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')
        is_color_recognition_enabled = 0
        fps = 24
        width = 720
        height = 480
        with open('test.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Time,Person_Number"                 
                writer.writerows([csv_line.split(',')])
		
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))#输出视频文件

        # input video
        cap = cv2.VideoCapture(0)#打开摄像头或者文件

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:#创建对象
            # 确定的输入输出 Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            #每个框表示图像中检测到特定对象的部分
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            #每个分数表示每个对象的置信水平
            #分数和类标签一起显示在结果图像上
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # 对于从输入视频中提取的所有帧
            last_person_number = person_flow = 0
            last_time = datetime.datetime.now()
            time_str = datetime.datetime.strftime(last_time,'%Y-%m-%d %H:%M:%S')
            #filename = time_str[:10]+'_'+time_str[11:13]+time_str[14:16]+time_str[17:19] + '.csv'
            filename = 'test.csv'

            while(cap.isOpened()):#如果摄像头打开
                ret, frame = cap.read()  #读取摄像头所拍摄内容              

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # 扩展维度为[1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # 运行模型，进行实际检测
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # 将信息文本插入视频帧
                font = cv2.FONT_HERSHEY_SIMPLEX
                #print('1')
				
                # 检测结果的可视化   
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
				
                #左上方的统计结果
                if(len(counting_mode) != 12):
                    now_person_number = 0
                    cv2.putText(input_frame, "person:0,person_flow:{}".format(person_flow), (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    now_person_number = int(counting_mode[11])
                    cv2.putText(input_frame, 'person:{},person_flow:{}'.format(now_person_number,person_flow), (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                    if now_person_number > last_person_number:
                        person_flow += (now_person_number - last_person_number)
                print('len(counting_mode)={}'.format(len(counting_mode)))#有人为12，无人为0
				
                #output_movie.write(input_frame)
                # print ("writing frame")
                #cv2.imshow('object counting',input_frame)#窗口输出
                c,d = cv2.imencode('.jpg', input_frame)
                frame = d.tobytes()
        
		        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

                if cv2.waitKey(1) & 0xFF == ord('q'):#q退出
                        break
                if stop_counting==1:
                    break
                
                
                last_person_number = now_person_number

                time1 = datetime.datetime.now()
                time_str = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
                interval_time = time1 - last_time
                print(interval_time)
                if( interval_time.seconds >= 6):
                        #print('判断不等于')
                        last_time = time1
                        with open(filename, 'a') as f:#
                                #print('追加打开文件')
                                
                                csv_line = "{},{}".format(time_str,person_flow)
                                
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])
                        person_flow = 0   
							
				
            cap.release()
            cv2.destroyAllWindows()
			
app = Flask(__name__)

@app.route('/')  #主页面
def index():  
    return render_template('index.html') 



@app.route('/new_data',methods=['GET','POST'])  #新数据
def new_data():
      
    return render_template('new_data.html')




@app.route('/last_data',methods=['GET','POST'])#历史数据  
def last_data():
    return render_template('last_data.html')



@app.route('/save')
def save():
    df= pd.read_csv("./test.csv")
    #print(df)
    #df_X = np.array(df.Time).tolist()
    #df_Y = np.array(df.Person_Number).tolist()
    #dic=zip(df_X,df_Y)
    #data={}
    filename=str(datetime.datetime.now().year)+'_'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)
    #data['Time']=df_X
    #data['Person_Number']=df_Y
    #with open("./static/"+filename+".json","w") as f:
    #    json.dump(data,f)
    df.to_csv("./static/"+filename+".csv")
    global file_road
    file_road = "./static/"+filename+".csv"
    return render_template('view.html',val1=time.time())  

@app.route('/counting_data',methods=['GET','POST'])  #数据采集页面
def counting_data():  
    #print("11111111111111111111111111111111111111111111111111111")
    global stop_counting
    if request.method == 'POST':
        temp = request.form.get('temp')
        stop_counting = 1
    return render_template('counting_data.html')

def gen(shexiangtou):
    while True:
        a,b = shexiangtou.read() 
        c,d = cv2.imencode('.jpg', b)
        frame = d.tobytes()
        
		# 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
     
@app.route('/video_feed')  # 这个地址返回视频流响应 3
def video_feed():   
    shexiangtou = cv2.VideoCapture(0)
    return Response(object_counting(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sucess_data',methods=['GET','POST'])  #数据采集成功
def sucess_data():  
    return render_template('sucess_data.html')



@app.route('/view',methods=['GET','POST'])  #视图
def view():
    global file_road
    file_road = "./static/"
    if request.method == 'POST':
        filename = request.form
        file_road = file_road+filename['myfile']
    plt.rcParams['font.sans-serif']='SimHei' #替换sans-serif字体
    plt.rcParams['axes.unicode_minus']=False #解决坐标轴负数的负号显示问题
    
    df = pd.read_csv(file_road)
    
    df_answer = pd.DataFrame(columns = ['start','person'])
    t = pd.DataFrame(columns = ['time'])
    st1 = df.Time[0]
    st2 = df.Time[len(df)-1]
    date_list_new = pd.date_range(st1,st2,freq='0.5H')
    
    for i in range(len(date_list_new)):
        st = "%s-" % (date_list_new[i].year)
        if date_list_new[i].month < 10:
            st =st + "0" + (str)(date_list_new[i].month) + "-"
        else:
            st = st + (str)(date_list_new[i].month) + "-"
        if date_list_new[i].day < 10:
            st =st + "0" + (str)(date_list_new[i].day) + " "
        else:
            st = st + (str)(date_list_new[i].day) + " "
        if date_list_new[i].hour < 10:
            st =st + "0" + (str)(date_list_new[i].hour) + ":"
        else:
            st = st + (str)(date_list_new[i].hour) + ":"
        if date_list_new[i].minute < 10:
            st =st + "0" + (str)(date_list_new[i].minute) + ":"
        else:
            st = st + (str)(date_list_new[i].minute) + ":"
        if date_list_new[i].second < 10:
            st =st + "0" + (str)(date_list_new[i].second)
        else:
            st = st + (str)(date_list_new[i].second)
        new=pd.DataFrame({'time' : st},index=[i])
        t=t.append(new,ignore_index=True)    
            
    t = pd.Series(t['time'].values,index=t.index)
    
    for i in range(len(date_list_new)-1):
        num = df.loc[(df['Time'] >= t[i]) & (df['Time'] < t[i+1]),['Person_Number']].sum()['Person_Number']
        new=pd.DataFrame({'start' :t[i],'person' : num},index=[i])
        df_answer=df_answer.append(new,ignore_index=True)
    num = df.loc[(df['Time'] >= t[len(date_list_new)-1]),['Person_Number']].sum()['Person_Number']
    new=pd.DataFrame({'start' :t[len(date_list_new)-1],'person' : num},index=[len(date_list_new)-1])
    df_answer=df_answer.append(new,ignore_index=True)
  
    df_answer['start'].str.split(' ')
    sta = df_answer['start'].str.split(' ',expand=True).apply(pd.Series)
    
    df_answer.plot(kind = 'bar')
    plt.title("客流量分析")
    plt.xlabel("时间")
    plt.ylabel("人流量")
    plt.rcParams['figure.figsize']=[24,16]
    for x1,y1 in enumerate(df_answer.person):
        plt.text(x1, y1+1, y1,ha='center',fontsize=10)
    plt.xticks(df_answer.index, sta[1],color='blue',rotation=60)
    plt.savefig('./static/pic1.jpg')
    
    df_answer.plot()
    plt.title("客流量分析")
    plt.xlabel("时间")
    plt.ylabel("人流量")
    plt.rcParams['figure.figsize']=[24,16]
    plt.xticks(df_answer.index, sta[1],color='blue',rotation=60)
    plt.savefig('./static/pic2.jpg')
    return render_template('view2.html',val1=time.time())
@app.route('/data',methods=['GET'])
def my_echart_data():
    df= pd.read_csv(file_road)
    print(df)
    df_X = np.array(df.Time).tolist()
    df_Y = np.array(df.Person_Number).tolist()
    #dic=zip(df_X,df_Y)
    data={}
    #filename=str(datetime.datetime.now().year)+'_'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)
    data['Time']=df_X
    data['Person_Number']=df_Y
    return jsonify(data)

if __name__ == '__main__':
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    http_server.serve_forever()
	