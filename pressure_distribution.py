import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid as integrate
import plotting

df_upper = pd.read_csv('Measurements  - Cp upper.csv', header=None)

# Extract the C_values into a numpy array
Cps_upper = df_upper.iloc[2:, 1:].astype(float).to_numpy()

df_lower = pd.read_csv('Measurements  - Cp lower.csv', header=None)

# Extract the C_values into a numpy array
Cps_lower = df_lower.iloc[2:, 1:].astype(float).to_numpy()

allangles = df_upper.iloc[2:, 0].astype(float).to_numpy()
angles = df_upper.iloc[2:33, 0].astype(float).to_numpy()

# x_c_upper = np.linspace(0, 1, Cps_upper.shape[1])
# x_c_lower = np.linspace(0, 1, Cps_lower.shape[1])

df_coords = pd.read_csv('Measurements  - Coordinates.csv', header=None)
xys_upper = df_coords.iloc[2:27, 1:3].astype(float).to_numpy()/100
y_c_upper = xys_upper[:,1]
x_c_upper = xys_upper[:,0]
xys_lower = df_coords.iloc[27:,1:3].astype(float).to_numpy()/100
y_c_lower = xys_lower[:,1]
x_c_lower = xys_lower[:,0]

def GetPlot_angle(angle):
	try:
		index = list(angles).index(angle)
	except:
		print("Invalid Angle")
		return
	plotting.plot_pressure_distribution(Cps_upper[index], Cps_lower[index], x_c_upper, x_c_lower, angle)

def GetPlot_index(index):
	plotting.plot_pressure_distribution(Cps_upper[index], Cps_lower[index], x_c_upper, x_c_lower, allangles[index])


# GetPlot_angle(0)
#GetPlot_angle(5)
# GetPlot_angle(10)
# GetPlot_angle(15)

def get_Cs(C_upper, C_lower, x_c_upper, x_c_lower, y_c_upper, y_c_lower):
	C_n = integrate(C_lower,x_c_lower)-integrate(C_upper,x_c_upper)
	C_m = -integrate(C_lower*x_c_lower,x_c_lower)+integrate(C_upper*x_c_upper,x_c_upper)
	C_t = integrate(C_upper,y_c_upper) + integrate(C_lower[::-1],y_c_lower[::-1])
	return C_n, C_m, C_t

Cns = []
Cms= []
Cts = []
k = 0
for i,j in zip(Cps_upper, Cps_lower):
	C_n , C_m, C_t = get_Cs(i,j,x_c_upper, x_c_lower, y_c_upper, y_c_lower)
	Cns.append(C_n)
	Cms.append(C_m)
	Cts.append(C_t)

Cns = Cns[2:31]
Cms = Cms[2:31]
Cts = Cts[2:31]

Cms = np.array(Cms)
Cns = np.array(Cns)
Cts = np.array(Cts)
Cm_quarter = Cms + 0.25*Cns
x_cps = -Cms/Cns

angles2 = angles[2:31]
Cls = Cns*np.cos(np.deg2rad(angles2)) - Cts*np.sin(np.deg2rad(angles2))
Cds = Cns*np.sin(np.deg2rad(angles2)) + Cts*np.cos(np.deg2rad(angles2))


# plt.scatter(angles2,x_cps)
# plt.show()
#PlotX_cps:
plotting.PlotCls(x_cps, angles2, r'$x_{cp}/c$')


index_to_plot_cp=7
#PlotCp:
#plotting.Plot2Cls(Cps_upper[index_to_plot_cp], Cps_lower[index_to_plot_cp], x_c_upper, '$C_{p,u}$', '$C_{p,l}$', r'x/c', x2s = x_c_lower, title_comment=r'$\alpha$ = ' + str(angles[index_to_plot_cp]) +r'$\degree$', invertedaxis=True, point_index=15, annotation_text="Separation", point_index2=17, annotation_text2="Transition", point_index3=19, annotation_text3="Reattachment")
index_to_plot_cp=13
plotting.Plot2Cls(Cps_upper[index_to_plot_cp], Cps_lower[index_to_plot_cp], x_c_upper, '$C_{p,u}$', '$C_{p,l}$', r'x/c', x2s = x_c_lower, title_comment=r'$\alpha$ = ' + str(angles[index_to_plot_cp]) +r'$\degree$', invertedaxis=True, point_index = 2, annotation_text="laminar separation bubble", point_index2=22, annotation_text2="turbulent separation")


#PlotCn:
plotting.PlotCls(Cns, angles2, r'$C_n$')

#PlotCt:
plotting.PlotCls(Cts, angles2, r'$C_t$')

#PlotCl
plotting.PlotCls(Cls, angles2, r'$C_l$')

#PlotCl&Cn
plotting.Plot2Cls(Cls, Cns, angles2, r'$C_l$', r'$C_n$')

#PlotCm
plotting.PlotCls(Cms,angles2, r'$C_m$')

#PlotCm_quarter
plotting.PlotCls(Cm_quarter, angles2, r'$C_{m,1/4}$')

#PlotCl
plotting.PlotCls(Cds, angles2, r'$C_d$')

#PlotCd over Cl
plotting.PlotCls(Cds, Cls, r'$C_d$', r'$C_l$')