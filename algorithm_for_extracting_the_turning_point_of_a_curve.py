# Algorithm for extracting the turning point of a curve.py
# Turning Point Extraction algorithms, improved based on the idea of the Douglas-Peucker algorithm.
# Copyright (c) 2026 Kun Li, Wuhan University
# Release Date: 2026-01-03


import numpy as np
import copy
np.set_printoptions(linewidth=200);


# Function to get the distance from a point to a line.
def get_Point2Line_distance(x, y, x1, y1, x2,y2):
    # xy represents the coordinates of a point, and x1y1 and x2y2 are two points that form a line segment.
    vector_line = np.array([x1-x2, y1-y2]);  #direction vector of the line segment
    vector_temp = np.array([x-x2, y-y2]);    #direction vector from a point to an endpoint of a line segment
    distance = np.sqrt(abs(np.sum(np.square(vector_temp)) - np.square(np.dot(vector_line, vector_temp)/np.linalg.norm(vector_line))));
    return distance


# Function to extract the most significant turning point of the curve based on the idea of DP algorithm.
def get_Turning_Point(x, y):
    #x and y must could be 1-D array.
    x = np.asarray(x).flatten();
    y = np.asarray(y).flatten();
    turningPoint_index = np.zeros(3, dtype=np.int32);
    turningPoint_index[0] = 0;
    turningPoint_index[1] = len(x)-1;

    distances = np.zeros(len(x), dtype=np.double);
    #Update the distances array
    for j in range(1,len(x)-1,1):
        j_down = turningPoint_index[0];  #0
        j_up = turningPoint_index[1];   #len(x)-1
        distances[j] = get_Point2Line_distance(x[j],y[j], x[j_down], y[j_down], x[j_up], y[j_up]);
    turningPoint_index[2] = np.argmax(distances);
    
    turningPoint_index = np.sort(turningPoint_index);

    return x[turningPoint_index], y[turningPoint_index]


# After extracting the turning point of the indicator data, the data is segmented based on the turning point. 
# Linear fitting is then performed on each segment, and the intersections are calculated as the optimized turning point.
def get_crossPoint_of_fitted_polyline(x, y, x_DP, isReturn_fit_params=False):
    # x and y must could be 1-D array.
    # x_DP is the x of the turning point and the two endpoints
    x = np.asarray(x).flatten();
    y = np.asarray(y).flatten();
    slopes = np.zeros(len(x_DP)-1,dtype=np.double);
    intercepts = np.zeros(len(x_DP)-1,dtype=np.double);
    crossPoint = np.zeros((len(x_DP)-2,2),dtype=np.double);

    for i in range(1, len(x_DP), 1):
        x_temp = x[(x>=x_DP[i-1])&(x<=x_DP[i])];
        y_temp = y[(x>=x_DP[i-1])&(x<=x_DP[i])];
        slope, intercept = np.polyfit(x_temp, y_temp, 1);
        slopes[i-1] = slope;
        intercepts[i-1] = intercept;

    for i in range(len(crossPoint)):
        D_temp = slopes[i+1] - slopes[i];
        if np.abs(D_temp) <= 1e-5:
            print(f'Warning: Two fitted lines are parallel, return the original turning point.');
            crossPoint[i,0] = x_DP[i];
            crossPoint[i,1] = y[np.argmax(x==x_DP[i])];
        else:
            # Calculate the x-coordinate of the intersection
            crossPoint[i,0] = (intercepts[i] - intercepts[i+1]) / (slopes[i+1] - slopes[i]);
            # Calculate the y-coordinate of the intersection point
            crossPoint[i,1] = (slopes[i+1]*intercepts[i] - slopes[i]*intercepts[i+1]) /  (slopes[i+1] - slopes[i]);    
    
    if isReturn_fit_params:
        return crossPoint, slopes, intercepts
    else:
        return  crossPoint



# For a certain indicator, find the date when its turning point (the critical point corresponding to developmental stage changes) was identified.
def get_result_of_TurningPoint_for_one_indicator(indicators:list, dates:list, filter_way=None, date_X_SE=[], 
                                                heuristic_way=None, heuristic_left_buffer = 30, heuristic_right_buffer=30, specific_day=None,
                                                true_date=None, true_date_label=None, 
                                                fit_polyline=False, fit_X_SE=[], isPrint=False):
    # indicators, time series indicators, list or array,
    # dates, the numbers indicating time, corresponding to the indicators, list or array
    # filter_way, parameter that controls the filtering method for indicators data, such as savgol (S-G) filtering. If set to None, no filtering is applied, and the original data is used directly.
    # date_X_SE, for the start and end times of the data to be taken, please enter two numbers, such as [0, 55]. If the list is empty, all data will be used by default.
    # true_date, the actual date (true value) of the key point
    # true_date_label, the text used to refer to the true_date; if not provided, it defaults to using 'true_date' as the true_date_label.
    # fit_polyline, parameter that controls whether to call get_crossPoint_of_fitted_polyline() to obtain the optimized turning point
    # fit_X_SE, when refining the turning point using the get_crossPoint_of_fitted_polyline(), the start and end times of the fitted range, currently support inputting two numbers, such as [0,55]. If it is an empty list, all data will be used by default.
    # heuristic algorithm.    heuristic_way, parameter controls the heuristic method. Currently, 'highest', 'specific_day' and None are supported. If it is None, no heuristic operation will be performed.
    #                                        If heuristic_way is a valid parameter, then date_X_SE and fit_X_SE are determined according to the heuristic method, and the input parameters are invalid.
    #                         heuristic_left_buffer,  heuristic_right_buffer: the range extending from the reference date to both sides. There is a default value, so input is optional.
    #                         specific_day, the reference date used when heuristic_way='specific_day'


    # Implementation: obtain the data, call the turning point extraction algorithm, and return the results.
    if isPrint:
        print('\nget_result_of_TurningPoint_for_one_indicator()...');
        print(f'indicators: {indicators}; ');
        print(f'dates: {dates};');
        print(f'filter_way: {filter_way}; true_date:{true_date};  true_date_label:{true_date_label}');
        print(f'date_X_SE: {date_X_SE}; fit_polyline: {fit_polyline}; fit_X_SE: {fit_X_SE}');
        print(f'heuristic_way: {heuristic_way};   heuristic_left_buffer:{heuristic_left_buffer},  heuristic_right_buffer:{heuristic_right_buffer},  specific_day:{specific_day}');
    
    result_dict = {};  #The variable to be returned. Save the data you want to return in this dictionary.

    # initialize date_X_SE, start_date, and end_date to the maximum valid data range
    [start_date, end_date] = [np.nanmin(dates), np.nanmax(dates)];
    if date_X_SE:
        if date_X_SE[0] is not None:
            if date_X_SE[0] < start_date:
                date_X_SE[0] = start_date;
        else:
            date_X_SE[0] = start_date;
        if date_X_SE[1] is not None:
            if date_X_SE[1] > end_date:
                date_X_SE[1] = end_date;
        else:
            date_X_SE[1] = end_date;
    else:
        date_X_SE = [start_date, end_date];
    if fit_X_SE:
        if fit_X_SE[0] is not None:
            if fit_X_SE[0] < start_date:
                fit_X_SE[0] = start_date;
        else:
            fit_X_SE[0] = start_date;
        if fit_X_SE[1] is not None:
            if fit_X_SE[1] > end_date:
                fit_X_SE[1] = end_date;
        else:
            fit_X_SE[1] = end_date;
    else:
        fit_X_SE = [start_date, end_date];

    
    # If filter_way is not None, perform one-dimensional filtering on the indicator data in the specified way.
    if filter_way is not None:
        indicators_original = indicators;
        from scipy import signal
        if "savgol" == filter_way:
            if isPrint:
                print(f'perform S-G filtering...');
            indicators = signal.savgol_filter(indicators, window_length=5, polyorder=3);
        else:
            print(f'Warning: the input filtering method is not supported yet,  filter_way: {filter_way};  still using the original indicator data');
    
    result_dict["Heuristic_day"] = None;
    if heuristic_way is not None:  # If heuristic_way is not None, perform heuristic optimization as instructed.
        if isPrint:
            print(f'performing heuristic optimization on the data range...');
        if 'highest' == heuristic_way:   
            highest_date = dates[np.nanargmax(indicators)];
            date_X_SE = [max(highest_date-heuristic_left_buffer, start_date), min(highest_date+heuristic_right_buffer, end_date)];
            fit_X_SE = [max(highest_date-heuristic_left_buffer, start_date), min(highest_date+heuristic_right_buffer, end_date)];
            result_dict["Heuristic_day"] = highest_date;
            if isPrint:
                print(f'heuristic_way: {heuristic_way}');
                print(f'highest_date: {highest_date}');
                print(f'[start_date, end_date]: {[start_date, end_date]}');
                print(f'date_X_SE: {date_X_SE};  fit_X_SE: {fit_X_SE}.');
        elif 'specific_day' == heuristic_way:
            date_X_SE = [max(specific_day-heuristic_left_buffer, start_date), min(specific_day+heuristic_right_buffer, end_date)];
            fit_X_SE = [max(specific_day-heuristic_left_buffer, start_date), min(specific_day+heuristic_right_buffer, end_date)];
            result_dict["Heuristic_day"] = specific_day;
            if isPrint:
                print(f'heuristic_way: {heuristic_way}');
                print(f'specific_day: {specific_day}');
                print(f'[start_date, end_date]: {[start_date, end_date]}');
                print(f'date_X_SE: {date_X_SE};  fit_X_SE: {fit_X_SE}.');
        else:
            print(f'Warning: an unknown heuristic method was inputted,  heuristic_way: {heuristic_way}; the originally specified range will still be used');
    
    indicator_data = indicators[(dates>=date_X_SE[0]) & (dates<=date_X_SE[1])];
    dates_data = dates[(dates>=date_X_SE[0]) & (dates<=date_X_SE[1])];

    if true_date_label is None:  # If not provided, 'true_date' defaults to true_date_label.
        true_date_label = "true_date";

    # Check if there are any NaNs in indicator_data; if there are, give a warning and automatically filter out the NaN data.
    indicator_data_isnan = np.isnan(indicator_data);
    if indicator_data_isnan.any():
        print(f'Warning: the indicator_data at position(s) [{np.argwhere(indicator_data_isnan).flatten()}] is/are NaN !!!');
        not_nan_args = np.argwhere(indicator_data_isnan == False);
        not_nan_args = not_nan_args.flatten();
        if isPrint:
            print(f'not_nan_args.flatten(): {not_nan_args}');
        [indicator_data, dates_data] = [indicator_data[not_nan_args], dates_data[not_nan_args],];



    # extract the turning point
    x_DP, indicator_DP = get_Turning_Point(dates_data, indicator_data);
    result_dict["Turning_date"] = int(x_DP[1]);
    result_dict["Turning_value"] = indicator_DP[1];
    result_dict["x_DP"] = x_DP;
    result_dict["indicator_DP"] = indicator_DP;
    detecting_date_delta = np.nan;
    if true_date is not None:  #If true_date is provided, then perform accuracy evaluation.
        detecting_date_delta = result_dict["Turning_date"] - true_date
    result_dict["detecting_date_delta"] = detecting_date_delta;
    if isPrint:
        print(f'Turning_date: {result_dict["Turning_date"]};  detecting_date_delta: {detecting_date_delta};  Turning_value: {result_dict["Turning_value"]}');
    
    if fit_polyline:
        # extract the optimized turning point
        # modify the start and end points of x_DP based on fit_X_SE
        x_DP_fit = copy.deepcopy(x_DP);
        indicator_DP_fit = copy.deepcopy(indicator_DP);
        dates_data_fit = dates[(dates>=fit_X_SE[0]) & (dates<=fit_X_SE[1])];
        indicator_data_fit = indicators[(dates>=fit_X_SE[0]) & (dates<=fit_X_SE[1])];
        x_DP_fit[0] = dates_data_fit[0]; 
        indicator_DP_fit[0] = indicator_data_fit[0];
        x_DP_fit[-1] = dates_data_fit[-1]; 
        indicator_DP_fit[-1] = indicator_data_fit[-1];
        [fit_crossPoint, fit_slopes, fit_intercepts] = get_crossPoint_of_fitted_polyline(dates, indicators, x_DP_fit, isReturn_fit_params=True);
        
        result_dict["fit_Turning_date"] = int(fit_crossPoint[0,0]);
        result_dict["fit_Turning_value"] = fit_crossPoint[0,1];
        result_dict["x_DP_fit"] = x_DP_fit;
        result_dict["indicator_DP_fit"] = indicator_DP_fit;
        result_dict["fit_crossPoint"] = fit_crossPoint;
        result_dict["fit_slopes"] = fit_slopes;
        result_dict["fit_intercepts"] = fit_intercepts;
        fit_detecting_date_delta = np.nan;
        if true_date is not None:  #If true_date is provided, then perform accuracy evaluation.
            fit_detecting_date_delta = result_dict["fit_Turning_date"] - true_date
        result_dict["fit_detecting_date_delta"] = fit_detecting_date_delta;
        if isPrint:
            print(f'fit_Turning_date: {result_dict["fit_Turning_date"]};  fit_detecting_date_delta: {fit_detecting_date_delta};  fit_Turning_value: {result_dict["fit_Turning_value"]}');

    return copy.deepcopy(result_dict)



# For a certain indicator, find the date when its turning point (the critical point corresponding to developmental stage changes) was identified, in an iteratively optimized version.
def get_iter_result_of_TurningPoint_for_one_indicator(indicators:list, dates:list, filter_way=None, date_X_SE=[],
                                                 heuristic_way=None, heuristic_left_buffer = 30, heuristic_right_buffer=30,  specific_day=None, 
                                                 iter_nums=3,  buffer_reduce_step = 5, 
                                                 true_date=None, true_date_label=None,
                                                 fit_polyline=False, fit_X_SE=[], isPrint=False):
    # indicators, time series indicators, list or array,
    # dates, the numbers indicating time, corresponding to the indicators, list or array,
    # filter_way, parameter that controls the filtering method for indicators data, such as savgol (S-G) filtering. If set to None, no filtering is applied, and the original data is used directly
    # date_X_SE, for the start and end times of the data to be taken, please enter two numbers, such as [0, 55]. If the list is empty, all data will be used by default
    # true_date, the actual date (true value) of the key point
    # true_date_label, the text used to refer to the true_date; if not provided, it defaults to using 'true_date' as the true_date_label
    # fit_polyline, parameter that controls whether to call get_crossPoint_of_fitted_polyline() to obtain the optimized turning point
    # fit_X_SE, when refining the turning point using the get_crossPoint_of_fitted_polyline(), the start and end times of the fitted range, currently support inputting two numbers, such as [0,55]. If it is an empty list, all data will be used by default.
    # heuristic algorithm.    heuristic_way, parameter controls the heuristic method. Currently, 'highest', 'specific_day' and None are supported. If it is None, no heuristic operation will be performed.
    #                                        If heuristic_way is a valid parameter, then date_X_SE and fit_X_SE are determined according to the heuristic method, and the input parameters are invalid.
    #                         heuristic_left_buffer,  heuristic_right_buffer: the range extending from the reference date to both sides. There is a default value, so input is optional.
    #                         specific_day, the reference date used when heuristic_way='specific_day'
    # iterative optimization.    iter_nums=3,  The number of iterative optimizations. The inputted heuristic parameters are only used at the first iteration.
    #                                          The subsequent heuristic method input is 'specific_day' taking the TP result from the previous iteration, buffer_size changes according to buffer_reduce_step
    #                            buffer_reduce_step = 5, the step size by which the buffer gradually decreases during iterative optimization. 
    #                                          Starting with a large buffer area helps to find global turning point, while a smaller buffer range later helps to accurately refine the positions of the truning point.
    
    # Implementation: achieved by calling get_result_of_TurningPoint_for_one_indicator()
    if isPrint:
        print('\nget_iter_result_of_TurningPoint_for_one_indicator()...');
        print(f'indicators: {indicators}; ');
        print(f'dates: {dates};');
        print(f'filter_way: {filter_way}; true_date:{true_date};  true_date_label:{true_date_label}');
        print(f'date_X_SE: {date_X_SE}; fit_polyline: {fit_polyline}; fit_X_SE: {fit_X_SE}');
        print(f'heuristic_way: {heuristic_way};   heuristic_left_buffer:{heuristic_left_buffer},  heuristic_right_buffer:{heuristic_right_buffer},  specific_day:{specific_day}');
        print(f'iter_nums:{iter_nums}; buffer_reduce_step:{buffer_reduce_step}');
    
    result_dict = {};  #The variable to be returned. Save the data you want to return in this dictionary.

    # initialize date_X_SE, start_date, and end_date to the maximum valid data range
    [start_date, end_date] = [np.nanmin(dates), np.nanmax(dates)];
    if date_X_SE:
        if date_X_SE[0] is not None:
            if date_X_SE[0] < start_date:
                date_X_SE[0] = start_date;
        else:
            date_X_SE[0] = start_date;
        if date_X_SE[1] is not None:
            if date_X_SE[1] > end_date:
                date_X_SE[1] = end_date;
        else:
            date_X_SE[1] = end_date;
    else:
        date_X_SE = [start_date, end_date];
    if fit_X_SE:
        if fit_X_SE[0] is not None:
            if fit_X_SE[0] < start_date:
                fit_X_SE[0] = start_date;
        else:
            fit_X_SE[0] = start_date;
        if fit_X_SE[1] is not None:
            if fit_X_SE[1] > end_date:
                fit_X_SE[1] = end_date;
        else:
            fit_X_SE[1] = end_date;
    else:
        fit_X_SE = [start_date, end_date];
    

    for i in range(iter_nums):
        if isPrint:
            print(f'iter: {i+1}');
        if 0==i: #The first time calling get_result_of_TurningPoint_for_one_indicator(), keep the default heuristic parameters.
            one_field_result_dict = get_result_of_TurningPoint_for_one_indicator(indicators=indicators, dates=dates, filter_way=filter_way, date_X_SE=date_X_SE, 
                                            heuristic_way=heuristic_way, heuristic_left_buffer=heuristic_left_buffer, heuristic_right_buffer=heuristic_right_buffer, specific_day=specific_day,
                                            fit_polyline=fit_polyline, fit_X_SE=fit_X_SE, true_date=true_date, true_date_label=true_date_label,  isPrint=isPrint);
        else:
            one_field_result_dict = get_result_of_TurningPoint_for_one_indicator(indicators=indicators, dates=dates, filter_way=filter_way, date_X_SE=date_X_SE, 
                                        heuristic_way='specific_day', specific_day=one_field_result_dict["Turning_date"], 
                                        heuristic_left_buffer=heuristic_left_buffer-buffer_reduce_step*i, heuristic_right_buffer=heuristic_right_buffer-buffer_reduce_step*i, 
                                        fit_polyline=fit_polyline, fit_X_SE=fit_X_SE, true_date=true_date, true_date_label=true_date_label, isPrint=isPrint);
        result_dict[f'iter_{i+1}'] = one_field_result_dict;
        if isPrint:
            print(f"result_dict['iter_{i+1}']: {result_dict[f'iter_{i+1}']}");
    
    result_dict[f'final'] = copy.deepcopy(one_field_result_dict);  #Give the final iteration result a 'final' tag and store it in the returned dict, making it easy to directly extract this important result using a fixed key.
    if isPrint:
        print(f"result_dict['final']: {result_dict[f'final']}");
    return copy.deepcopy(result_dict)