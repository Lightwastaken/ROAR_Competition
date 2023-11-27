import json

with open('PIDconfig.json') as json_file:
    data = json.load(json_file)
prev_key = 15
currspeed = int(70)
strcurrspeed = str(int(currspeed))
controller_values = data["Throttle_Controller"]
Speed_array = []
for key in controller_values:
    Speed_array.append(int(key))
Speed_array.append(currspeed)
Speed_array.sort()
try:
    K_Values_Determinant = Speed_array[Speed_array.index(currspeed) + 1]
except:
    K_Values_Determinant = Speed_array[Speed_array.index(currspeed)]

print(K_Values_Determinant)
