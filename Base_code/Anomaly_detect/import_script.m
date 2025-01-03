

p=10
c=3






%% Import data from spreadsheet
% Script for importing data from the following spreadsheet:
%
%    Workbook: D:\PS!\dg models\Data_generation\diffusion added\raw_data\C0\0_130\C0_P130_D180_1_CO2.xlsx
%    Worksheet: Sheet2
%
% Auto-generated by MATLAB on 31-Oct-2019 16:48:18

%% Setup the Import Options
opts = spreadsheetImportOptions("NumVariables", 9);

% Specify sheet and range
opts.Sheet = "Sheet2";
opts.DataRange = "A2:I10802";

% Specify column names and types
opts.VariableNames = ["Time", "CO2_Zone_1", "CO2_Zone_2", "CO2_Zone_3", "CO2_Zone_4", "CO2_Zone_5", "CO2_Zone_6", "flightNo", "tailNo"];
opts.SelectedVariableNames = ["Time", "CO2_Zone_1", "CO2_Zone_2", "CO2_Zone_3", "CO2_Zone_4", "CO2_Zone_5", "CO2_Zone_6", "flightNo", "tailNo"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "categorical", "categorical"];
opts = setvaropts(opts, [8, 9], "EmptyFieldRule", "auto");

% Import the data
file_name=strcat('D:\PS!\dg models\Data_generation\diffusion added\raw_data\C',num2str(c),'\',num2str(c),'_',num2str(p),'\C',num2str(c),'_P0',num2str(p),'_D180_1_CO2.xlsx');

tbl = readtable(file_name, opts, "UseExcel", false);

%% Convert to output type
Time = tbl.Time;
CO2_Zone_1 = tbl.CO2_Zone_1;
CO2_Zone_2 = tbl.CO2_Zone_2;
CO2_Zone_3 = tbl.CO2_Zone_3;
CO2_Zone_4 = tbl.CO2_Zone_4;
CO2_Zone_5 = tbl.CO2_Zone_5;
CO2_Zone_6 = tbl.CO2_Zone_6;
flightNo = tbl.flightNo;
tailNo = tbl.tailNo;

%% Clear temporary variables
clear opts tbl



%% Import data from spreadsheet
% Script for importing data from the following spreadsheet:
%
%    Workbook: D:\PS!\dg models\Data_generation\diffusion added\raw_data\C0\0_130\C0_P130_D180_1_TPH.xlsx
%    Worksheet: Sheet1
%
% Auto-generated by MATLAB on 31-Oct-2019 16:48:23

%% Setup the Import Options
opts = spreadsheetImportOptions("NumVariables", 13);

% Specify sheet and range
opts.Sheet = "Sheet1";
opts.DataRange = "A2:M181";

% Specify column names and types
opts.VariableNames = ["Time2", "CO2min_Zone_1", "CO2min_Zone_2", "CO2min_Zone_3", "CO2min_Zone_4", "CO2min_Zone_5", "CO2min_Zone_6", "flightNo2", "t_sim", "p_sim", "h_sim", "tailNo2", "pas_cnt"];
opts.SelectedVariableNames = ["Time2", "CO2min_Zone_1", "CO2min_Zone_2", "CO2min_Zone_3", "CO2min_Zone_4", "CO2min_Zone_5", "CO2min_Zone_6", "flightNo2", "t_sim", "p_sim", "h_sim", "tailNo2", "pas_cnt"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "categorical", "double", "double", "double", "categorical", "double"];
opts = setvaropts(opts, [8, 12], "EmptyFieldRule", "auto");

% Import the data
file_name=strcat('D:\PS!\dg models\Data_generation\diffusion added\raw_data\C',num2str(c),'\',num2str(c),'_',num2str(p),'\C',num2str(c),'_P0',num2str(p),'_D180_1_TPH.xlsx');
tbl = readtable(file_name, opts, "UseExcel", false);


%% Convert to output type
Time2 = tbl.Time2;
CO2min_Zone_1 = tbl.CO2min_Zone_1;
CO2min_Zone_2 = tbl.CO2min_Zone_2;
CO2min_Zone_3 = tbl.CO2min_Zone_3;
CO2min_Zone_4 = tbl.CO2min_Zone_4;
CO2min_Zone_5 = tbl.CO2min_Zone_5;
CO2min_Zone_6 = tbl.CO2min_Zone_6;
flightNo2 = tbl.flightNo2;
t_sim = tbl.t_sim;
p_sim = tbl.p_sim;
h_sim = tbl.h_sim;
tailNo2 = tbl.tailNo2;
pas_cnt = tbl.pas_cnt;

%% Clear temporary variables
clear opts tbl






