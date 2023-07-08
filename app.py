from flask import Flask, render_template,Response,request,redirect
from face_jetson import face_detect_frames, setData1
from face_jetson_local_video import face_detect_localvideo
from pcl_jetson import crossing_line,setData2
from pcl_jetson_local import crossing_line_local
from time_2 import time_frames,setData3
from time_2_local import time_local
from add_staff_jetson import add_new_staff, capt
from screen_video_gender import take_screen_video_gender
from gender_local_cam import recogenderlocal
from gender_webcam import recogender_webcam

#Recognition
from local_cam import recoloc
#Age
from age_local_cam import recoage
#URL
from reco_image import urlrecognition
from url_image import urldetection
import json
from os.path import exists
from delete_emp import delete_em
from os import mkdir, listdir
from compreface_python_sdk_main.webcam_demo.test_collection import runTC
from compreface_python_sdk_main.webcam_demo.test_recognition import run
import threading
urlde=""
urlre=""
emp=""
app = Flask(__name__)

username = ""
password = ""

def log():
	global username
	global password
	username=""
	password=""

def is_user_authenticated():
	global username
	global password
	if username=='admin' and password=='admin':
		return True
	else:
		return False
	
@app.errorhandler(404)
def page_not_found(error):
    # redirect the user to the home path
    return redirect('/home')
	
@app.route('/logout')
def logout():
	log()
	is_user_authenticated()
	return redirect('/')
#Home
@app.route('/home')
def index():
	if not is_user_authenticated():
		return redirect('/')
	else:
		return render_template('./index.html')
#Login
@app.route('/', methods=['GET','POST'])
def login():
	global username
	global password
	if not is_user_authenticated():
		if request.method == 'POST':		
			username= request.form['username']
			password= request.form['password']
			if is_user_authenticated():
				return redirect('/home')
			else:
				return redirect('/')
		return render_template('./login.html')
	else:
		return redirect('/home')
	

@app.route('/deleteempl', methods=['GET','POST'])
def deleteempl():
	global emp
	if not is_user_authenticated():
		return redirect('/')
	else:
		if request.method == 'POST':		
			emp= request.form['emp']
			delete_em(emp)
		return render_template('./delete_emp.html')


'''
@app.route('/', methods=['GET','POST'])
def sliders():
	if request.method =='POST':
		x1 = request.form['slider_x1']
		x2 =request.form['slider_x2']
		y1 = request.form['slider_y1']
		y2 = request.form['slider_y2']
		print("x1 = ", x1)
		print("x2 = ", x2)
		print("y1 = ", y1)
		print("y2 = ", y2)
		return x1,x2,y1,y2 
'''

#Video + html + face detect + Redirect slider data to face_jetson
@app.route('/face', methods=['GET','POST'])
def face():
	if not is_user_authenticated():
		return redirect('/')
	else:
		if request.method == 'POST':		
			x1= request.values.get('slider_x1')
			x2= request.values.get('slider_x2')
			y1= request.values.get('slider_y1')
			y2= request.values.get('slider_y2')
			setData1(x1,x2,y1,y2)
		return render_template('./face.html')
@app.route('/app1')
def app1():
	return Response(face_detect_frames() ,mimetype='multipart/x-mixed-replace; boundary=image')

@app.route('/facehtml', methods=['GET','POST'])
def facehtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		if request.method == 'POST':
			x1= request.values.get('slider_x1')
			x2= request.values.get('slider_x2')
			y1= request.values.get('slider_y1')
			y2= request.values.get('slider_y2')
			setData1(x1,x2,y1,y2)
		return render_template('./face_local.html')

@app.route('/faceapp')
def faceapp():
	return Response(face_detect_localvideo() ,mimetype='multipart/x-mixed-replace; boundary=image')

#Video + html + People crossing line
@app.route('/line', methods=['GET','POST'])
def line():
	if not is_user_authenticated():
		return redirect('/')
	else:
		if request.method == 'POST':		
			x1= request.values.get('sli_x1')
			setData2(x1)
		return render_template('./line.html')

@app.route('/app2')
def app2():
	return Response(crossing_line() ,mimetype='multipart/x-mixed-replace; boundary=image')

@app.route('/linehtml', methods=['GET','POST'])
def linehtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		if request.method == 'POST':
			x1= request.values.get('sli_x1')
			setData2(x1)
		return render_template('./line_local.html')

@app.route('/lineapp')
def lineapp():
	return Response(crossing_line_local() ,mimetype='multipart/x-mixed-replace; boundary=image')

#Video + html + Time _jetson
@app.route('/time', methods=['GET','POST'])
def time():
	if not is_user_authenticated():
		return redirect('/')
	else:	
		if request.method == 'POST':		
			x1= request.values.get('sl_x1')
			x2= request.values.get('sl_x2')
			y1= request.values.get('sl_y1')
			y2= request.values.get('sl_y2')
			setData3(x1,x2,y1,y2)
		return render_template('./time.html')

#video + time
@app.route('/app3')
def app3():
	return Response(time_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/timehtml', methods=['GET','POST'])
def timehtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		if request.method == 'POST':
			x1= request.values.get('sl_x1')
			x2= request.values.get('sl_x2')
			y1= request.values.get('sl_y1')
			y2= request.values.get('sl_y2')
			setData3(x1,x2,y1,y2)
		return render_template('./time_local.html')

#video + time
@app.route('/timeapp')
def timeapp():
	return Response(time_local(),mimetype='multipart/x-mixed-replace; boundary=frame')


#Add staff
@app.route('/add_staff')
def add_staff():
	if not is_user_authenticated():
		return redirect('/')
	else:
		if request.args.get('status') == '0':
			nom = request.args.get('nom')
			if exists('staffs/' + nom):
				files = listdir('staffs/' + nom)
				if len(files) >= 5:
					j = {'nom': nom}
					with open('staffs/' + nom + '/' + nom + '.json', 'w+') as file:
						file.writelines(json.dumps(j))
					# start functions
					t1 = threading.Thread(target=runTC, args=('staffs/' + nom + '/' + files[[i.endswith('jpg') for i in listdir('staffs/' + nom)].index(True)], nom))
					t1.start()
					t2 = threading.Thread(target=run, args=('staffs/' + nom,))
					t2.start()
					"""print(runTC('staffs/' + nom + '/' + files[[i.endswith('jpg') for i in listdir('staffs/' + nom)].index(True)], nom))
					print(run('staffs/' + nom))"""
			return redirect('/add_staff')
		elif request.args.get('status') == '1':
			nom = request.args.get('nom')
			capt(nom)
			return redirect('/add_staff')
		else:
			return render_template('./add_staff.html')

@app.route('/app4')
def app4():
	return Response(add_new_staff(),mimetype='multipart/x-mixed-replace; boundary=image')

#Test Staff

#Recognition
@app.route('/reco')
def reco():
	return render_template('./recognition.html')

@app.route('/recoapp')
def recoapp():	
	return Response(recoloc() ,mimetype='multipart/x-mixed-replace; boundary=frame')

#Detection Age
@app.route('/recoagehtml')
def recoagehtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		return render_template('./age.html')

@app.route('/recoageapp')
def recoageapp():
	return Response(recoage() ,mimetype='multipart/x-mixed-replace; boundary=frame')

#Gender
@app.route('/descriptionrecogenderhtml',methods=['GET', 'POST'])
def descriptionrecogenderhtml():
	global urlde
	if not is_user_authenticated():
		return redirect('/')
	else:
		if request.method == 'POST':
			urlde = request.form['urlde']
		return render_template('./description_gender.html')


@app.route('/descriptionrecognitionhtml',methods=['GET', 'POST'])
def descriptionrecognitionhtml():
	global urlre
	if not is_user_authenticated():
		return redirect('/')
	else:
		if request.method == 'POST':
			urlre = request.form['urlre']
		return render_template('./description_recognition.html')

@app.route('/descriptionfacehtml')
def descriptionfacehtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		return render_template('./description_face.html')

@app.route('/descriptionlinehtml')
def descriptionlinehtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		return render_template('./description_line.html')
	


@app.route('/descriptiontimehtml')
def descriptiontimehtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		return render_template('./description_time.html')

@app.route('/recogenderhtml')
def recogenderhtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		return render_template('./gender.html')

@app.route('/recogenderapp')
def recogenderapp():
	return Response(recogender_webcam() ,mimetype='multipart/x-mixed-replace; boundary=frame')

#Gender URL Image
@app.route('/urldetecthtml')
def urldetecthtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		return render_template('./urldetect_gender.html')

@app.route('/urldetectapp')
def urldetectapp():
	global urlde
	print(urlde)
	return Response(urldetection(urlde) ,mimetype='multipart/x-mixed-replace; boundary=frame')

#Recognition URL Image
@app.route('/urlrecohtml')
def urlrecohtml():
	if not is_user_authenticated():
		return redirect('/')
	else:
		return render_template('./urldetect_recognition.html')

@app.route('/urlrecoapp')
def urlrecoapp():
	global urlre
	print(urlre)
	return Response(urlrecognition(urlre) ,mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=2204, threaded=True)

