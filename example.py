# example for extracting the turning point of a curve
# Author: Kun Li, Wuhan University
# Release Date: 2026-01-03

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import algorithm_for_extracting_the_turning_point_of_a_curve as TP


# set the font
# import matplotlib as mpl
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['Arial'];  #'Times New Roman',  'Arial'; 


# read the typical curve
print("read the typical curve...")
df = pd.read_excel('data.xlsx',  sheet_name=f'example');
x_sample = np.asarray(df["x_sample"]);
y_sample = np.asarray(df["y_sample"]);
print(f'x_sample: {x_sample}');
print(f'y_sample: {y_sample}');
plt.plot(x_sample, y_sample, "-", label="ideal data");
plt.scatter(x_sample, y_sample, label="ideal sample");   

y_sample_with_noize = np.asarray(df["y_sample_with_noize"]);
print(f'y_sample_with_noize: {y_sample_with_noize}');
plt.scatter(x_sample, y_sample_with_noize, color="C6",  label="sample with noize");  
plt.plot(x_sample, y_sample_with_noize, color="C6",  linestyle='dashed');  

true_date_example = 94.5;
plt.plot([true_date_example, true_date_example],[np.nanmin(y_sample_with_noize), np.nanmax(y_sample_with_noize)], linestyle='dotted', color = 'red' , linewidth = 1, alpha =1,  label='true day'); 
plt.title("ideal_data and sample_data_with_noize");
plt.legend();
plt.show()


# plot the max point
print("plot the max point...")
plt.plot(x_sample, y_sample_with_noize, '-o', color="black",  label="sequential data");  
plt.plot(x_sample[np.nanargmax(y_sample_with_noize)], np.nanmax(y_sample_with_noize), '*', markersize=12, color = 'C0', label = 'max point');
indicator_data_SGed = signal.savgol_filter(y_sample_with_noize, window_length=5, polyorder=3);
plt.plot(x_sample, indicator_data_SGed, "--", color="C6",  label="SG filtered");  
plt.plot(x_sample[np.nanargmax(indicator_data_SGed)], np.nanmax(indicator_data_SGed), '*', markersize=10, color = 'C1', label = 'SG - max point');
plt.title("Max and SG-Max");
plt.legend();
plt.show()





# plot the result of TP(Turning Point) and OTP(Optimized Turning Point)
print("plot the result of TP(Turning Point) and OTP(Optimized Turning Point)...")
filter_way = None;
result_dict = TP.get_result_of_TurningPoint_for_one_indicator(y_sample_with_noize, x_sample, filter_way=filter_way, date_X_SE=[None, None], 
                                             heuristic_way="highest", heuristic_left_buffer=40, heuristic_right_buffer=40, specific_day=None,
                                             true_date=true_date_example, true_date_label="true date", fit_polyline=True, fit_X_SE=None);

plt.plot(x_sample, y_sample_with_noize, '-o', color="black",  label="sequential data");  
plt.plot([result_dict["Heuristic_day"], result_dict["Heuristic_day"]], [np.nanmin(y_sample_with_noize)-0.1, np.nanmax(y_sample_with_noize)+0.1],  linestyle='dotted', color = 'black' , linewidth = 1, alpha =0.7, label = 'heuristic position');
plt.plot(result_dict["x_DP"][1], result_dict["indicator_DP"][1], 'o-',  markersize=8,  color = 'limegreen', label= 'key point');
plt.plot(result_dict["x_DP"], result_dict["indicator_DP"], '-',  alpha=0.8,  color='limegreen', );
plt.plot(result_dict["x_DP"].take([0,-1]), result_dict["indicator_DP"].take([0,-1]), '-', alpha=0.8,  color='deepskyblue', label= 'range');
plt.title("Turning Point Extraction");
plt.legend();
plt.show()



plt.plot(x_sample, y_sample_with_noize, '-o', color="black",  label="sequential data");  
plt.plot([result_dict["Heuristic_day"], result_dict["Heuristic_day"]], [np.nanmin(y_sample_with_noize)-0.1, np.nanmax(y_sample_with_noize)+0.1],  linestyle='dotted', color = 'black' , linewidth = 1, alpha =0.7, label = 'heuristic position');
plt.plot(result_dict["x_DP"][1], result_dict["indicator_DP"][1], 'o',  markersize=8,  color = 'limegreen', label= 'key point');
plt.plot(result_dict["x_DP"].take([0,-1]), result_dict["indicator_DP"].take([0,-1]), '-', alpha=0.8,  color='deepskyblue', label= 'range');
plt.plot([result_dict["x_DP_fit"][0], result_dict["fit_crossPoint"][0,0], result_dict["x_DP_fit"][2]], [result_dict["x_DP_fit"][0]*result_dict["fit_slopes"][0]+result_dict["fit_intercepts"][0], result_dict["fit_crossPoint"][0,1], result_dict["x_DP_fit"][2]*result_dict["fit_slopes"][1]+result_dict["fit_intercepts"][1]],
         '-', color='orange', linewidth=1.5, alpha=0.8, label= 'curve fit');
plt.plot([result_dict["fit_crossPoint"][0,0],], [result_dict["fit_crossPoint"][0,1]], 'P', markersize=8, color='darkorange', linewidth=1, alpha=1, label= 'OTP');
plt.title("Optimized Turning Point Extraction");
plt.legend();
plt.show()





# plot the process of iteration of TP
print("plot the process of iteration of TP...")
iter_nums = 4;
x_lim_down = 50;
result_dict = TP.get_iter_result_of_TurningPoint_for_one_indicator(y_sample_with_noize, x_sample, filter_way=None, date_X_SE=[x_lim_down, None], 
                                             heuristic_way="highest", heuristic_left_buffer=40, heuristic_right_buffer=40, specific_day=None,
                                             iter_nums=iter_nums, buffer_reduce_step=7,
                                             true_date=true_date_example, true_date_label="true date", fit_polyline=True, fit_X_SE=None);
full_result_dict = result_dict;

for i in range(iter_nums):
    result_dict = full_result_dict[f'iter_{i+1}'];
    plt.plot(x_sample[x_sample>x_lim_down], y_sample_with_noize[x_sample>x_lim_down], '-o', color="black",  label="sequential data");  
    plt.plot([result_dict["Heuristic_day"], result_dict["Heuristic_day"]], [np.nanmin(y_sample_with_noize)-0.1, np.nanmax(y_sample_with_noize)+0.1],  linestyle='dotted', color = 'black' , linewidth = 1, alpha =0.7, label = 'heuristic position');
    plt.plot(result_dict["x_DP"][1], result_dict["indicator_DP"][1], 'o',  markersize=8,  color = 'limegreen', label= 'key point');
    plt.plot(result_dict["x_DP"].take([0,-1]), result_dict["indicator_DP"].take([0,-1]), '-', alpha=0.8,  color='deepskyblue', label= 'range');
    plt.plot([result_dict["x_DP_fit"][0], result_dict["fit_crossPoint"][0,0], result_dict["x_DP_fit"][2]], [result_dict["x_DP_fit"][0]*result_dict["fit_slopes"][0]+result_dict["fit_intercepts"][0], result_dict["fit_crossPoint"][0,1], result_dict["x_DP_fit"][2]*result_dict["fit_slopes"][1]+result_dict["fit_intercepts"][1]],
                '-', color='orange', linewidth=1.5, alpha=0.8, label= 'curve fit');
    plt.plot([result_dict["fit_crossPoint"][0,0],], [result_dict["fit_crossPoint"][0,1]], 'P', markersize=8, color='darkorange', linewidth=1, alpha=1, label= 'OTP');
    plt.title(f'Turning Point Extraction: iter_{i+1}');
    plt.legend();
    plt.show()

