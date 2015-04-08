# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:00:09 2015

@author: Ying
"""

def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180


def add_feature(data):
    data['Aspect2']=data.Aspect.map(r)
    
    data['Ele_minus_VDtHyd'] = data.Elevation-data.Vertical_Distance_To_Hydrology
         
    data['Ele_plus_VDtHyd'] = data.Elevation+data.Vertical_Distance_To_Hydrology
     
    data['Distanse_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology']**2+data['Vertical_Distance_To_Hydrology']**2)**0.5
     
    data['Hydro_plus_Fire'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Fire_Points']
     
    data['Hydro_minus_Fire'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Fire_Points']
     
    data['Hydro_plus_Road'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Roadways']
     
    data['Hydro_minus_Road'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Roadways']
     
    data['Fire_plus_Road'] = data['Horizontal_Distance_To_Fire_Points']+data['Horizontal_Distance_To_Roadways']
     
    data['Fire_minus_Road'] = data['Horizontal_Distance_To_Fire_Points']-data['Horizontal_Distance_To_Roadways']
    
    
   # data['Soil']=0
    #for i in range(1,41):
    #    data['Soil']=data['Soil']+i*data['Soil_Type'+str(i)]
     
    
    #data['Wilderness_Area']=0
    #for i in range(1,5):
     #    data['Wilderness_Area']=data['Wilderness_Area']+i*data['Wilderness_Area'+str(i)]

