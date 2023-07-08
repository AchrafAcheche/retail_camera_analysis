#
# This file is part of Vizy 
#
# All Vizy source code is provided under the terms of the
# GNU General Public License v2 (http://www.gnu.org/licenses/gpl-2.0.html).
# Those wishing to use Vizy source code, software and/or
# technologies under different licensing terms should contact us at
# support@charmedlabs.com. 
#

from threading import Thread
from vizy import Vizy
from kritter import Camera, Kritter, Kvideo, Kslider,render_detected, import_config
import kritter
from tfliteDetectorV2 import TFLiteDetector
import time  
import datetime
import subprocess
import os 
import cv2
import numpy as np
import schedule
from kimagedetector import render_roi_image,render_detected_rect,render_detected
#from perspective import Perspective
from dash_devices.dependencies import Output, State
import dash_html_components as html
import dash_bootstrap_components as dbc
import imagezmq

CONSTS_FILE = "face_detect_consts.py"
APP_DIR = os.path.dirname(os.path.realpath(__file__))
MEDIA_DIR = os.path.join(APP_DIR, "media")
#sender=imagezmq.ImageSender(connect_to="tcp://172.18.4.121:5555")

class TFliteExample:

    def __init__(self):
        self.kapp = Vizy()
        #consts_filename = os.path.join(APP_DIR, CONSTS_FILE) 
        #self.config_consts = import_config(consts_filename, self.kapp.etcdir, ["WIDTH", "PADDING", "GRAPHS", "MAX_RECORDING_DURATION", "START_SHIFT", "MIN_RANGE", "PLAY_RATE", "UPDATE_RATE", "FOCAL_LENGTH", "BG_AVG_RATIO", "BG_CNT_FINAL", "EXT_BUTTON_CHANNEL", "DEFAULT_CAMERA_SETTINGS", "DEFAULT_CAPTURE_SETTINGS", "DEFAULT_PROCESS_SETTINGS", "DEFAULT_ANALYZE_SETTINGS"])     
        
        # Instantiate Vizy's camera and camera stream
        self.camera = kritter.Camera(hflip=True, vflip=True)
        self.stream = self.camera.stream()
        self.id = self.kapp.new_id("TFliteExample")
        style = kritter.default_style
        control_style = style
        style = style.copy()
        #self.camera.framerate = 5
        
        self.sensitivity=0.50
        self.x=749
        self.y=216                                                
        self.x1=16
        self.y1=25
        self.total="0"
        self.run="stopped"
        self.txt = ""
        self.dic={}
        self.ag=0
        self.gen=0
        self.eth=0
        self.bs=0
        self.video = Kvideo(width=self.camera.resolution[0], overlay=True)
        
        control_style = style
        style = style.copy()
        style['control_width'] = 0.1
        
        #--------------------------Settings--------------------------------------------------------------------
        self.stop = kritter.Kbutton(name=[Kritter.icon("power-off"), "Stop"], disabled=False, value=self.run)
        self.start = kritter.Kbutton(name=[Kritter.icon("play-circle"), "Start"],  disabled=False, value = self.run)
        self.rein = kritter.Kbutton(name=[Kritter.icon("chevron-circle-up")," Reinitialize"], disabled=False, value = self.run)
        self.setROI = kritter.Kbutton(name=[Kritter.icon("plus"), "Set Zone"], disabled=False)
        #self.perspective = Perspective(self.video, self.config_consts.FOCAL_LENGTH, self.camera.getmodes()[self.camera.mode], style=control_style)
        self.stop.append(self.start)
        self.stop.append(self.rein)
        self.stop.append(self.setROI)
        
        #---------------------------Server Analytics-------------------------------------------------------
        self.title =  kritter.Ktext(name="Server Analytics ", style=style)
        self.serverip =  kritter.KtextBox(name="Server IP", placeholder="Enter Server IP ", style=style)
        self.title_enable =  kritter.Ktext(name="Enable Analytics For: ", style={"margin-left": f"{style['horizontal_padding']}px", "margin-right": f"{style['horizontal_padding']}px"})
        self.age = kritter.Kcheckbox(name="Age", value= False, style=style)
        self.gender = kritter.Kcheckbox(name="Gender", value= False, style=style)
        self.ethnicity = kritter.Kcheckbox(name="Ethnicity", value= False, style=style)
        self.bosket = kritter.Kcheckbox(name="Bosket", value= False, style=style)
        self.age.append(self.gender)
        self.age.append(self.ethnicity)
        self.age.append(self.bosket)
    
        #--------------------------------------- Settings Camera-------------------------------------------------------------------------------------------
        self.brightness = kritter.Kslider(name="Preview brightness", value=self.camera.brightness, mxs=(0, 100, 1), format=lambda val: '{}%'.format(val), style=control_style)
        sensitivity_c = Kslider(name="Sensitivity", value=self.sensitivity*100, mxs=(10, 90, 1), format=lambda val: f'{int(val)}%',style=control_style)
        
        #--------------------------------ROI Settings-------------------------------------------------------------------------------------------------------
        
        x_b = Kslider(name="Right", value=self.x, mxs=(384, 768, 1), format=lambda val: f'{int(val)}',  style=control_style)
        x1_b = Kslider(name="Left", value=self.x1, mxs=(0, 384, 1), format=lambda val: f'{int(val)}',  style=control_style)
        y_b = Kslider(name="Down", value=self.y, mxs=(216, 432, 1), format=lambda val: f'{int(val)}',  style=control_style)
        y1_b = Kslider(name="Up", value=self.y1, mxs=(0, 216, 1), format=lambda val: f'{int(val)}',  style=control_style)
        
        #-----------------------------------------------------Div-------------------------------------------------------------------------------------------
        
        controls = [x_b, x1_b, y_b, y1_b]
        self.collapse = dbc.Collapse(dbc.Card(controls, style={"margin-left": f"{style['horizontal_padding']}px", "margin-right": f"{style['horizontal_padding']}px"}), id=Kritter.new_id())
        
        
        self.div1 = dbc.Collapse(dbc.Card([ self.title, self.serverip, self.title_enable, self.age], style={"margin-left": f"{style['horizontal_padding']}px", "margin-right": f"{style['horizontal_padding']}px"}), is_open= True, id=self.kapp.new_id())
        
        #-------------------------------------------- Set application layout-------------------------------------------------------------------------------------------
        
        self.kapp.layout = html.Div([html.Div([self.video]), self.stop, self.div1 ,self.collapse, self.brightness, sensitivity_c], id=Kritter.new_id())
        
        #-------------------------------------------- Callbacks----------------------------------------------------------------------------------------------------
        @self.start.callback([State(self.collapse.id, "is_open")])
        def func(is_open):
            return self.set_more(False)
        
        @self.stop.callback([State(self.collapse.id, "is_open")])
        def func(is_open):
            return self.set_more(False)
        
        @self.rein.callback([State(self.collapse.id, "is_open")])
        def func(is_open):
            return self.set_more(False)
        
        
        @self.start.callback([State(self.div1.id, "is_open")])
        def func(is_open):
            self.run="started"
            print(self.run)
            self.dic={"server_ip":self.txt,"age":self.ag, "gender": self.gen, "ethnicity": self.eth, "bosket": self.bs}
            print(self.dic)
            #sender.send_image(filename+'.jpg',image)
            return self.set_more_serv(False)
        
        @self.stop.callback([State(self.div1.id, "is_open")])
        def func(is_open):
            self.run="stopped"
            print(self.run)
            return self.set_more_serv(not is_open)
        
        @self.rein.callback([State(self.div1.id, "is_open")])
        def func(is_open):
            self.run="rein"
            print(self.run)
            return self.set_more_serv(False)
        
        @self.setROI.callback([State(self.div1.id, "is_open")])
        def func(is_open):
            return self.set_more_serv(False)
            
        
        @self.age.callback()
        def func(value):
            if value == True:
                self.ag=1
            elif value == False:
                self.ag=0
            print(self.dic)
        
        @self.gender.callback()
        def func(value):
            if value == True:
                self.gen=1
            elif value == False:
                self.gen=0
            print(self.dic)
            
        @self.ethnicity.callback()
        def func(value):
            if value ==  True:
                self.eth=1
            elif value == False:
                self.eth=0
            print(self.dic)
            
        @self.bosket.callback()
        def func(value):
            if value == True:
                self.bs=1
            elif value == False:
                self.bs=0
            print(self.dic)
            
        @self.setROI.callback([State(self.collapse.id, "is_open")])
        def func(is_open):
            return self.set_more(not is_open)
                
        @self.serverip.callback()
        def func(val):
            if val:
                self.txt = val
               
        
            
        # Callback for sensitivity slider
        @sensitivity_c.callback()
        def func(value):
            # Update sensitivity value, convert from %
            self.sensitivity = value/100
            
        @x_b.callback()
        def func(value):
            # Update x value
            self.x = value
            
        @y_b.callback()
        def func(value):
            # Update y value
            self.y = value
        
        @x1_b.callback()
        def func(value):
            # Update x value
            self.x1 = value
            
        @y1_b.callback()
        def func(value):
            # Update y value
            self.y1 = value
        
        @self.brightness.callback()
        def func(value):
            self.camera.brightness = value

        
            
        # Instantiate TensorFlow Lite detector
        self.tflite = TFLiteDetector()

        # Start processing thread
        self.run_process = True
        Thread(target=self.process).start()

        # Run Vizy server, which blocks.
        self.kapp.run()
        self.run_process = False
    
   

    def set_more(self, val):
        return [Output(self.collapse.id, "is_open", val)]
    def set_more_serv(self, val):
        return [Output(self.div1.id, "is_open", val)]
    
    


    
    # Frame processing thread
    def process(self):
        
        while self.run_process:
            
            # Get frame
            frame = self.stream.frame()[0]
            #frame1 = self.perspective.transform(frame)
            # Run detection
            t0 = time.time()
            
            dets = self.tflite.detect(frame, self.sensitivity,  self.x, self.y, self.x1, self.y1, self.run)[0]
            total=str(self.tflite.detect(frame, self.sensitivity,  self.x, self.y, self.x1, self.y1, self.run)[1])
            #print(time.time()-t0)
            
                       # If we detect something...
            if dets is not None:
                self.kapp.push_mods(render_detected(self.video.overlay, dets,total, self.x, self.y, self.x1, self.y1))
                
            # Push frame to the video window in browser.
            if self.run == "stopped":
                self.video.overlay.draw_clear(id='tag')
                self.video.overlay.draw_text(230, 380, "Detection is stopped", font=dict(family="sans-serif", size=30, color="red"), fillcolor="black",padding=3, xanchor="left", id='tag')
                
            elif self.run == "rein":
                self.video.overlay.draw_clear(id='tag')
                self.video.overlay.draw_text(230, 380, "Counter is reinitialized", font=dict(family="sans-serif", size=30, color="green"), fillcolor="black",padding=3, xanchor="left", id='tag')    
                    
            elif self.run == "started":
                self.video.overlay.draw_clear(id='tag')
            self.video.push_frame(frame)
            
    
if __name__ == '__main__':
    TFliteExample()
   

