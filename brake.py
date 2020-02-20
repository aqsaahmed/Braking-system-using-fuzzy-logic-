# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:18:15 2019

@author: user
"""

import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt

# Creating variables

x_pedal = np.arange(0, 101, 1)
x_speed = np.arange(0, 101, 1)
y_brake = np.arange(0, 101, 1)

# Creating membership functions

pedal_low = mf.trimf(x_pedal, [0, 0, 50])
pedal_med = mf.trimf(x_pedal, [0, 50, 100])
pedal_hig = mf.trimf(x_pedal, [50, 100, 100])

speed_low = mf.trimf(x_pedal, [0, 0, 60])
speed_med = mf.trimf(x_pedal, [20, 50, 80])
speed_hig = mf.trimf(x_pedal, [40, 100, 100])

brake_poor = mf.trimf(y_brake, [0, 0, 100])
brake_strong = mf.trimf(y_brake, [0, 100, 100])

# Data visualization

fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize =(6, 10))

ax0.plot(x_pedal, pedal_low, 'r', linewidth = 2, label = 'Low')
ax0.plot(x_pedal, pedal_med, 'g', linewidth = 2, label = 'Middle')
ax0.plot(x_pedal, pedal_hig, 'b', linewidth = 2, label = 'High')
ax0.set_title('Pedal Pressure')
ax0.legend()

ax1.plot(x_speed, speed_low, 'r', linewidth = 2, label = 'Low')
ax1.plot(x_speed, speed_med, 'g', linewidth = 2, label = 'Middle')
ax1.plot(x_speed, speed_hig, 'b', linewidth = 2, label = 'High')
ax1.set_title('Vehicle Speed')
ax1.legend()

ax2.plot(y_brake, brake_poor, 'r', linewidth = 2, label = 'Weak')
ax2.plot(y_brake, brake_strong, 'b', linewidth = 2, label = 'Strong')
ax2.set_title('Brake')
ax2.legend()

plt.tight_layout()

input_pedal = 40
input_speed = 75

# Calculation of membership degrees

pedal_fit_low = fuzz.interp_membership(x_pedal, pedal_low, input_pedal)
pedal_fit_med = fuzz.interp_membership(x_pedal, pedal_med, input_pedal)
pedal_fit_hig = fuzz.interp_membership(x_pedal, pedal_hig, input_pedal)

speed_fit_low = fuzz.interp_membership(x_speed, speed_low, input_speed)
speed_fit_med = fuzz.interp_membership(x_speed, speed_med, input_speed)
speed_fit_hig = fuzz.interp_membership(x_speed, speed_hig, input_speed)

# Creating rules

rule1 = np.fmin(pedal_fit_med, brake_strong)
rule2 = np.fmin(np.fmin(pedal_fit_hig, speed_fit_hig), brake_strong)
rule3 = np.fmin(np.fmax(pedal_fit_low, speed_fit_low), brake_poor)
rule4 = np.fmin(pedal_fit_low, brake_poor)

# Creating union sets

out_strong = np.fmax(rule1, rule2)
out_poor = np.fmax(rule3, rule4)

# Data Visualization

brake0 = np.zeros_like(y_brake)

fig, ax0 = plt.subplots(figsize = (7, 4))
ax0.fill_between(y_brake, brake0, out_poor, facecolor = 'r', alpha = 0.7)
ax0.plot(y_brake, brake_poor, 'r', linestyle = '--')
ax0.fill_between(y_brake, brake0, out_strong, facecolor = 'g', alpha = 0.7)
ax0.plot(y_brake, brake_strong, 'g', linestyle = '--')
ax0.set_title('Brake output')

plt.tight_layout()

# Clarifier

out_brake = np.fmax(out_poor, out_strong)

defuzzified  = fuzz.defuzz(y_brake, out_brake, 'centroid')

result = fuzz.interp_membership(y_brake, out_brake, defuzzified)

# Result

print("(Brake) Output Value:", defuzzified)

# Data visualization

fig, ax0 = plt.subplots(figsize=(7, 4))

ax0.plot(y_brake, brake_poor, 'b', linewidth = 0.5, linestyle = '--')
ax0.plot(y_brake, brake_strong, 'g', linewidth = 0.5, linestyle = '--')
ax0.fill_between(y_brake, brake0, out_brake, facecolor = 'Orange', alpha = 0.7)
ax0.plot([defuzzified , defuzzified], [0, result], 'k', linewidth = 1.5, alpha = 0.7)
ax0.set_title('Rinsing with centre of gravity')

plt.tight_layout()