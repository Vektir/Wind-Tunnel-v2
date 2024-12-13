import numpy as np
import pandas as pd
from scipy.integrate import trapezoid as integrate
import matplotlib.pyplot as plt



# y_totals = np.array([0, 12, 21, 27, 33, 39, 45, 51, 57, 63, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 156, 162, 168, 174, 180, 186, 195, 207, 219])
# y_statics = np.array([43.5, 55.5, 67.5, 79.5, 91.5, 103.5, 115.5, 127.5, 139.5, 151.5, 163.5, 175.5])

coordinates = pd.read_csv('Measurements  - Coordinates.csv', header=None)
pd_p_statics = pd.read_csv("Measurements  - Wake pressures static.csv", header=None)
pd_p_totals = pd.read_csv("Measurements  - Wake pressures total.csv", header=None)
pd_pitot_tube = pd.read_csv("Measurements  - Pitot Tube .csv", header=None)

y_totals = coordinates.iloc[2:-2,5].astype(float).to_numpy()/1000
y_statics = coordinates.iloc[2:14,8].astype(float).to_numpy()/1000

p_statics = pd_p_statics.iloc[2:,4:].astype(float).to_numpy() #cringe values = cringe outcomes
p_totals = pd_p_totals.iloc[2:,4:].astype(float).to_numpy()

U_infs = pd_p_statics.iloc[2:,3].astype(float).to_numpy()# based on the beginning of the wind tunnel
alternative_U_infs = np.array([21.26117004, 21.15951222, 21.13382053, 21.18976044, 21.2250153, 
                 21.18855753, 21.20138499, 21.28018372, 21.02714999, 21.21740841, 
                 21.20859703, 21.11263639, 21.18133867, 21.08646278, 20.96645377, 
                 21.26460715, 21.27684379, 21.11516848, 21.21820927, 21.1159739, 
                 21.18713189, 21.19234877, 21.14052432, 21.14253539, 21.25365115, 
                 21.16263554, 21.26365108, 21.29721645, 21.30799409, 21.36896472, 
                 21.25245084, 21.37845406, 21.33062926, 21.20899047, 21.16440207, 
                 21.21661332, 21.24547511, 21.10561083, 21.30308139, 21.27429775, 
                 21.21460757, 21.17525654, 21.14106622, 21.18731053, 21.23625964, 
                 21.03937545, 21.07655751, 21.23986617, 21.16239137, 21.21260163, 
                 21.15917386, 21.2558878, 21.10964285, 21.18972051, 21.16802084, 
                 21.29628876, 21.15917386, 21.20377325, 21.13784546])



p_infs_pitot = pd_pitot_tube.iloc[3:,2].astype(float).to_numpy()

angles = pd_p_statics.iloc[2:, 0].astype(float).to_numpy()

densities = data = np.array([1.178, 1.178, 1.178, 1.177, 1.177, 1.177, 1.177, 1.177, 1.177, 
				1.177, 1.177, 1.177, 1.177, 1.177, 1.177, 1.177, 1.176, 1.176, 
				1.177, 1.176, 1.176, 1.176, 1.176, 1.176, 1.176, 1.176, 1.176, 
				1.176, 1.176, 1.176, 1.176, 1.175, 1.175, 1.175, 1.175, 1.175, 
				1.175, 1.175, 1.175, 1.175, 1.175, 1.175, 1.175, 1.175, 1.175, 
				1.175, 1.175, 1.175, 1.175, 1.175, 1.175, 1.175, 1.175, 1.175, 
				1.175, 1.175, 1.175, 1.175, 1.175])


qs = p_totals - np.mean(p_statics, axis=1)[:, np.newaxis] # i mean it is vague enough for me to do whatever tf i want

#print(p_totals[10])
#print(qs[10])

velocities = np.sqrt(2*qs/densities[:, np.newaxis])


drag1s = []
for v,U_inf,rho in zip(velocities, alternative_U_infs,densities):
	Drag_1 = integrate(rho*(U_inf-v), y_totals)
	drag1s.append(Drag_1)

drag2s = []
for p_stat, p_infs_pitot in zip(p_statics, p_infs_pitot):
	Drag_2 = integrate(p_infs_pitot-p_stat,  y_statics)
	drag2s.append(Drag_2)


drag1s = np.array(drag1s)
drag2s = np.array(drag2s)
totaldrag = drag1s + drag2s

import plotting


index = 7
#plotting.PlotWakeRake(velocities[index], y_totals, angles[index])

index2 = 18+24
#plotting.PlotWakeRake(velocities[index2], y_totals, angles[index2])
#plotting.PlotArbitrary(drag1s[:]/.4,angles[:])
plotting.PlotCls(drag1s[:]/.4, angles[:], r'$C_{D1}$')



coefficients = np.polyfit(angles, drag1s[:]/.4, 2)

x2s = np.linspace(min(angles), max(angles), 20)
y2s = np.polyval(coefficients, x2s)


coefficientsboth = np.polyfit(angles, totaldrag[:]/.4, 2)
y2sboth = np.polyval(coefficientsboth, x2s)
plotting.Plot2Cls(totaldrag[:]/.4, y2sboth, angles[:], r'$C_D$', '$C_{D-total}$',scatter=True,  x2s=x2s)

plotting.Plot2Cls(drag1s[:]/.4, y2s, angles[:], r'$C_D$', '$C_{D-poly}$',scatter=True,  x2s=x2s)


plotting.PlotCls(velocities[index], y_totals, r'$V_{wake} [m/s]$', 'y [mm]', title_comment=r'$\alpha$ = ' + str(angles[index]) + r'$\degree$')



