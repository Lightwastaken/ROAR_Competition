import matplotlib.pyplot as plt

from PyGame_Viewer2 import PyGameViewer2
import roar_py_carla
import roar_py_interface
import carla
import numpy as np
import asyncio
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, Any, List

#     streamlit run C:\Users\roar\Desktop\Roar_Monza\ROAR_Competition\competition_code\Streamlit.py [ARGUMENTS]
class Streamlit:
    def __init__(self):
        self.arraydepth = None
        self.arraysec = None

    def startup(self):
        st.title('scuffed data dashboard')
        attempt_num = st.number_input('insert attempt num in logs')
        f = open(r'C:\Users\roar\Desktop\Roar_Monza\ROAR_Competition\competition_code\testlogs.txt', 'r')
        lines = f.readlines()
        f.close()
        if attempt_num:
            line_num_ofdepth = attempt_num + 2
            line_num_of_sec = attempt_num + 1
            self.arraydepth = lines[int(line_num_ofdepth) - 1]
            self.arraydepth = self.arraydepth[1 : len(self.arraydepth) - 2]
            self.arraydepth = self.arraydepth.split(',')
            print("depth" + str(self.arraydepth))
            self.arraysec = lines[int(line_num_of_sec) - 1]
            self.arraysec = self.arraysec[1 : len(self.arraysec) - 2]
            self.arraysec = self.arraysec.split(',')
            print("sec: " + str(self.arraysec))
            chart_data = pd.DataFrame(
                {
                    "depth": self.arraydepth,
                    "sec": self.arraysec,
                }
            )
            st.line_chart(chart_data, x="sec", y="depth", color=["#0000FF"])


streamStream = Streamlit()
streamStream.startup()
