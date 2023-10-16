import matplotlib.pyplot as plt

from PyGame_Viewer2 import PyGameViewer2
import roar_py_carla
import roar_py_interface
import carla
import numpy as np
import asyncio
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

st.title('scuffed data dashboard')
# depthNum = st.number_input('insert depth line')
# secondsNum = st.number_input('insert second line')
# f = open(r'C:\Users\roar\Desktop\Roar_Monza\ROAR_Competition\competition_code\testlogs.txt', 'r')
# if depthNum is not None and secondsNum is not None:
#     depth = [6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16]
#     seconds = [31.791580070947415, 31.791580070947415, 31.791580070947415, 31.791580070947415, 31.791580070947415, 31.665760583028828, 31.665760583028828, 31.665760583028828, 31.665760583028828, 31.66585714255912, 31.665646475890068, 31.66585714255912, 31.665587002371968, 31.66585714255912, 31.665587002371968, 31.66585714255912, 31.665587002371968, 31.66585714255912, 31.665587002371968, 31.66585714255912, 31.665728181941994, 31.665728181941994, 31.665728181941994, 31.665728181941994, 32.02883360080919, 31.665728181941994, 32.66056194666397, 31.665728181941994, 32.66056194666397, 31.22742206021679, 32.66056194666397, 31.22742206021679, 32.66056194666397, 31.22742206021679, 32.66056194666397, 32.24777332352241, 32.24777332352241, 32.24777332352241, 32.24777332352241, 32.38496355920813, 32.2242425217773, 32.38496355920813, 31.92622370280169, 32.38496355920813, 31.449802978623094, 32.38496355920813, 31.30473729042633, 32.38496355920813, 31.261089547937484, 32.38496355920813, 31.150838150431994, 32.38496355920813, 30.99427493538111, 30.99427493538111, 30.99427493538111, 30.99427493538111, 31.176953075942585, 30.99427493538111, 31.519128019757748, 30.99427493538111, 31.5399169838379, 30.99427493538111, 31.72280280845182, 30.99427493538111, 31.768759954497813, 30.99427493538111, 31.768759954497813, 31.563253829673158, 31.563253829673158, 31.563253829673158, 31.451591041778983, 31.563253829673158, 31.451591041778983, 31.563253829673158, 31.343778058515674, 31.563253829673158, 31.343778058515674, 31.563253829673158, 31.15765098080939, 31.563253829673158, 30.818732722922128, 31.563253829673158, 30.609704828840787, 31.563253829673158, 30.63261814311851, 30.63261814311851, 30.63261814311851, 30.57626328803678, 30.63261814311851, 30.57626328803678, 31.082635157265376, 30.57626328803678, 31.261303054171986, 30.57626328803678, 31.261303054171986, 30.57626328803678, 31.261303054171986, 30.57626328803678, 31.261303054171986, 31.08867536119672, 31.08867536119672, 31.08867536119672, 31.08867536119672, 31.145201896739124, 30.94046462180999, 31.145201896739124, 30.76603859222165, 31.145201896739124, 30.76603859222165, 31.145201896739124, 30.49548619839467, 31.145201896739124, 30.49548619839467, 31.145201896739124, 30.826839467694725, 30.826839467694725, 30.826839467694725, 30.674790236639396, 30.826839467694725, 30.674790236639396, 31.169922698135533, 30.674790236639396, 31.30588976060687, 30.674790236639396, 31.526862605027116, 30.674790236639396, 31.526862605027116, 31.6176910124833, 31.6176910124833, 31.6176910124833, 31.532066801313565, 31.6176910124833, 31.532066801313565, 31.773899310463626, 31.532066801313565, 31.773899310463626, 31.532066801313565, 31.817130983896913, 31.532066801313565, 31.817130983896913, 31.617107549733372, 31.617107549733372, 31.617107549733372, 31.617107549733372, 31.658341382643062]
#     seconds = np.array(secondsNum)
#     depth = np.array(depthNum)
# if depth is not None and seconds is not None:
#     chart_data = pd.DataFrame(
#         {
#             "col1": seconds,
#             "col2": depth,
#         }
#     )
#     st.line_chart(chart_data, x="col1", y="col2", color="#ffaa00", index=[0])
# seconds = [6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16]
# depth = [31.791580070947415, 31.791580070947415, 31.791580070947415, 31.791580070947415, 31.791580070947415, 31.665760583028828, 31.665760583028828, 31.665760583028828, 31.665760583028828, 31.66585714255912, 31.665646475890068, 31.66585714255912, 31.665587002371968, 31.66585714255912, 31.665587002371968, 31.66585714255912, 31.665587002371968, 31.66585714255912, 31.665587002371968, 31.66585714255912, 31.665728181941994, 31.665728181941994, 31.665728181941994, 31.665728181941994, 32.02883360080919, 31.665728181941994, 32.66056194666397, 31.665728181941994, 32.66056194666397, 31.22742206021679, 32.66056194666397, 31.22742206021679, 32.66056194666397, 31.22742206021679, 32.66056194666397, 32.24777332352241, 32.24777332352241, 32.24777332352241, 32.24777332352241, 32.38496355920813, 32.2242425217773, 32.38496355920813, 31.92622370280169, 32.38496355920813, 31.449802978623094, 32.38496355920813, 31.30473729042633, 32.38496355920813, 31.261089547937484, 32.38496355920813, 31.150838150431994, 32.38496355920813, 30.99427493538111, 30.99427493538111, 30.99427493538111, 30.99427493538111, 31.176953075942585, 30.99427493538111, 31.519128019757748, 30.99427493538111, 31.5399169838379, 30.99427493538111, 31.72280280845182, 30.99427493538111, 31.768759954497813, 30.99427493538111, 31.768759954497813, 31.563253829673158, 31.563253829673158, 31.563253829673158, 31.451591041778983, 31.563253829673158, 31.451591041778983, 31.563253829673158, 31.343778058515674, 31.563253829673158, 31.343778058515674, 31.563253829673158, 31.15765098080939, 31.563253829673158, 30.818732722922128, 31.563253829673158, 30.609704828840787, 31.563253829673158, 30.63261814311851, 30.63261814311851, 30.63261814311851, 30.57626328803678, 30.63261814311851, 30.57626328803678, 31.082635157265376, 30.57626328803678, 31.261303054171986, 30.57626328803678, 31.261303054171986, 30.57626328803678, 31.261303054171986, 30.57626328803678, 31.261303054171986, 31.08867536119672, 31.08867536119672, 31.08867536119672, 31.08867536119672, 31.145201896739124, 30.94046462180999, 31.145201896739124, 30.76603859222165, 31.145201896739124, 30.76603859222165, 31.145201896739124, 30.49548619839467, 31.145201896739124, 30.49548619839467, 31.145201896739124, 30.826839467694725, 30.826839467694725, 30.826839467694725, 30.674790236639396, 30.826839467694725, 30.674790236639396, 31.169922698135533, 30.674790236639396, 31.30588976060687, 30.674790236639396, 31.526862605027116, 30.674790236639396, 31.526862605027116, 31.6176910124833, 31.6176910124833, 31.6176910124833, 31.532066801313565, 31.6176910124833, 31.532066801313565, 31.773899310463626, 31.532066801313565, 31.773899310463626, 31.532066801313565, 31.817130983896913, 31.532066801313565, 31.817130983896913, 31.617107549733372, 31.617107549733372, 31.617107549733372, 31.617107549733372, 31.658341382643062]
# fig = plt.Figure()
# plt.plot(seconds, depth)
# st.pyplot()