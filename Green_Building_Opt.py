# -*- coding: utf-8 -*-
"""
- Created within the Framework of a masterthesis @
faculty of Process Engineering, Energy and Mechanical Systems
of the Cologne University of Applied Sciences -

-> Main Programm Green_Building_Opt.py
Data Import, Timeseries Simulation, Multiobjective Optimization &
Performance Estimation of a solar-hydrogen energy system for
existing residential buildings - Germany
@author: marius bartkowski
@coauthors: Henrik Naß, Fares Aoun, Jannis Grunenberg

Contact:
marius.bartkowski@magenta.de or
marius.bartkowski@smail.th-koeln.de
"""

# Path & Data Handling
import os
import glob

# The usual suspects
import sys
import datetime as dt
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Math
from scipy import signal

# PV Lib functions
from pvlib.ivtools.sdm import fit_cec_sam

# Console
from tqdm import tqdm
import warnings

# Classes & Global Functions
from energy_system import solar_rooftiles
from energy_system import electrical_energy_storage
from energy_system import heat_pump_ashp
from energy_system import h2_boiler
from energy_system import warm_water_storage_mixed
from energy_system import energy_system_operating_strategy
from energy_system import electrical_vehicle_ems, electrical_vehicle_normal

# importing electrical vehicle function
from electrical_vehicle import ev_ems_connection_profile, ev_normal_connection_profile

from parametrization import heat_flux_temperature_difference
from parametrization import q_loss_regression
from parametrization import q_loss_specific_din12831
from parametrization import overall_costs
from parametrization import autarky_rate as autarky_rate_calculation
from parametrization import co2_emission_timeline

# importing Paralization function
# this library is used to paralize running fucntions on all CPU cores
from joblib import Parallel, delayed

# importing own multi-objective function
from optimization import optimize_multi_objective

# %% importing plotting functions
from plotting_functions import plot_Pareto_Fronts, plotting_Sensitivity, plot_PCP
# function performs Sensitivity analysis
from plotting_functions import sensivity_analysis

# %%Program Controls


plt.close('all')  # close all plots.

"____Programm_Controll_____"

# If True, the programm carries out a multi-objective optimization. If False only a timerseries simulation for the design specified by the Decision Variables below in the row below and the component dicts is carried out.
optimization_mode = False
# If True, the energy system including a fuel cell chp component is simulated
combined_heat_and_power = False
# If true, give output results also as table
output_results_as_table = False
# If True there is a Sensitivity analysis to be performed
Sensitivity_analysis = False
plot_results = False


# %%
# Get Current Working directory and Input_parameter filepath
path = os.getcwd()
filepath_input_parameters = (path + "/Input/Input_parameters.xlsx")


"____Own system design for Non-Optimization Mode____"
df_owndesign_parameters = pd.read_excel(
    filepath_input_parameters, sheet_name="Own Design", index_col=0, usecols="A:B")

# System Design without CHP
if combined_heat_and_power == False:
    df_owndesign_parameters_without_CHP = df_owndesign_parameters.iloc[2: 11]
    own_design = df_owndesign_parameters_without_CHP.to_numpy(
        dtype="float").reshape((1, 9))

elif combined_heat_and_power == True:                                           # System Design with CHP
    df_owndesign_parameters_with_CHP = df_owndesign_parameters.iloc[13: 23]
    own_design = df_owndesign_parameters_with_CHP.Values.to_numpy(
        dtype="float").reshape((1, 10))

else:
    pass


"_____General_Parameters______"
df_general_parameters = pd.read_excel(
    filepath_input_parameters, sheet_name="General Parameters", index_col=0, usecols="A:B")

cp_w = df_general_parameters["Values"].loc[df_general_parameters.index[1]]
rho_w = df_general_parameters["Values"].loc[df_general_parameters.index[2]]
ho_h2 = df_general_parameters["Values"].loc[df_general_parameters.index[3]]
hu_h2 = df_general_parameters["Values"].loc[df_general_parameters.index[4]]
n = df_general_parameters["Values"].loc[df_general_parameters.index[5]]
year = df_general_parameters["Values"].loc[df_general_parameters.index[6]]
scaling_factor_electric_energy_demand = df_general_parameters[
    "Values"].loc[df_general_parameters.index[7]]
hu_ng = df_general_parameters["Values"].loc[df_general_parameters.index[8]]
ho_ng = df_general_parameters["Values"].loc[df_general_parameters.index[9]]
# rho_ng = df_general_parameters["Values"].loc[df_general_parameters.index[10]]
# rho_h2 = df_general_parameters["Values"].loc[df_general_parameters.index[11]]

"_____Optimization_Parameters_______"
df_opti_parameters = pd.read_excel(
    filepath_input_parameters, sheet_name="Optimization Parameters", index_col=0, usecols="A:B")

df_opti_parameters_specs = df_opti_parameters.iloc[2: 7]
df_opti_parameters_specs.columns = df_opti_parameters.iloc[1]
dict_opti_parameters = df_opti_parameters_specs.to_dict()
optimization_specs = dict_opti_parameters["optimization_specs"]


"_____Energy_System_Parameters_- Technical____"
df_tech_parameters = pd.read_excel(
    filepath_input_parameters, sheet_name="Technical Parameters", index_col=0, usecols="A:B")

df_tech_parameters_building = df_tech_parameters.iloc[2: 13]
df_tech_parameters_building.columns = df_tech_parameters.iloc[1]
dict_tech_parameters_building = df_tech_parameters_building.to_dict()
building = dict_tech_parameters_building["building"]

df_tech_parameters_pvt = df_tech_parameters.iloc[15: 37]
df_tech_parameters_pvt.columns = df_tech_parameters.iloc[14]
dict_tech_parameters_pvt = df_tech_parameters_pvt.to_dict()
hybrid_solar_rooftiles = dict_tech_parameters_pvt["hybrid_solar_rooftiles"]

df_tech_parameters_fan = df_tech_parameters.iloc[39:42]
df_tech_parameters_fan.columns = df_tech_parameters.iloc[38]
dict_tech_parameters_fan = df_tech_parameters_fan.to_dict()
fan_heat_pump = dict_tech_parameters_fan["fan_heat_pump"]

df_tech_parameters_elstorage = df_tech_parameters.iloc[44:47]
df_tech_parameters_elstorage.columns = df_tech_parameters.iloc[43]
dict_tech_parameters_elstorage = df_tech_parameters_elstorage.to_dict()
electrical_storage = dict_tech_parameters_elstorage["electrical_storage"]

df_tech_parameters_hp = df_tech_parameters.iloc[49:52]
df_tech_parameters_hp.columns = df_tech_parameters.iloc[48]
dict_tech_parameters_hp = df_tech_parameters_hp.to_dict()
heat_pump = dict_tech_parameters_hp["heat_pump"]

df_tech_parameters_wwstorage = df_tech_parameters.iloc[54:70]
df_tech_parameters_wwstorage.columns = df_tech_parameters.iloc[53]
dict_tech_parameters_wwstorage = df_tech_parameters_wwstorage.to_dict()
ww_storage = dict_tech_parameters_wwstorage["ww_storage"]

df_tech_parameters_h2boiler = df_tech_parameters.iloc[72:75]
df_tech_parameters_h2boiler.columns = df_tech_parameters.iloc[71]
dict_tech_parameters_h2boiler = df_tech_parameters_h2boiler.to_dict()
boiler_specs_h2 = dict_tech_parameters_h2boiler["boiler_specs_h2"]

df_tech_parameters_chp = df_tech_parameters.iloc[77:82]
df_tech_parameters_chp.columns = df_tech_parameters.iloc[76]
dict_tech_parameters_chp = df_tech_parameters_chp.to_dict()
chp_specs = dict_tech_parameters_chp["chp_specs"]

df_tech_parameters_ngboiler = df_tech_parameters.iloc[84:87]
df_tech_parameters_ngboiler.columns = df_tech_parameters.iloc[83]
dict_tech_parameters_ngboiler = df_tech_parameters_ngboiler.to_dict()
boiler_specs_ng = dict_tech_parameters_ngboiler["boiler_specs_ng"]

df_tech_parameters_ev = df_tech_parameters.iloc[89:100]
df_tech_parameters_ev.columns = df_tech_parameters.iloc[88]
dict_tech_parameters_ev = df_tech_parameters_ev.to_dict()
ev_df = dict_tech_parameters_ev["ev_specs"]


"_____Energy_System_Parameters_- Ecological____"
df_ecological_parameters = pd.read_excel(
    filepath_input_parameters, sheet_name=5, index_col=0, usecols="A:B")
df_ecological_parameters_CO2 = df_ecological_parameters.iloc[2:3]
df_ecological_parameters_CO2.columns = df_ecological_parameters.iloc[1]
dict_ecological_parameters_CO2 = df_ecological_parameters_CO2.to_dict()
specs_ecological = dict_ecological_parameters_CO2["specs_ecological"]

"_____Energy_System_Parameters_-_Economical____"
df_eco_parameters = pd.read_excel(
    filepath_input_parameters, sheet_name="Economical Parameters", index_col=0, usecols="A:E", keep_default_na=False)

df_eco_parameters_specs = df_eco_parameters.iloc[2:47]
values_capex_hp = df_eco_parameters_specs.loc["CAPEX HP"].values.flatten(
).tolist()
values_capex_battery = df_eco_parameters_specs.loc["CAPEX Battery"].values.flatten(
).tolist()
values_capex_ww = df_eco_parameters_specs.loc["CAPEX Water Storage"].values.flatten(
).tolist()
values_capex_h2boiler = df_eco_parameters_specs.loc["CAPEX H2 Boiler"].values.flatten(
).tolist()
values_capex_h2chp = df_eco_parameters_specs.loc["CAPEX H2 CHP"].values.flatten(
).tolist()
df_eco_parameters_specs = df_eco_parameters_specs.drop(index=["CAPEX HP", "CAPEX Battery", "CAPEX Water Storage", "CAPEX H2 Boiler", "CAPEX H2 CHP"], columns=[
                                                       "Values (b)", "Reference Year", "Cost Reduction to 2023 [%]"])
df_eco_parameters_specs.rename(
    columns={"Values (a)": "specs_eco"}, inplace=True)
dictof_eco_parameters_specs = df_eco_parameters_specs.to_dict()
specs_eco = dictof_eco_parameters_specs["specs_eco"]
specs_eco["CAPEX HP"] = values_capex_hp
specs_eco["CAPEX Battery"] = values_capex_battery
specs_eco["CAPEX Water Storage"] = values_capex_ww
specs_eco["CAPEX H2 Boiler"] = values_capex_h2boiler
specs_eco["CAPEX H2 CHP"] = values_capex_h2chp

df_renov_parameters_specs = df_eco_parameters.iloc[49: 58]
df_renov_parameters_specs = df_renov_parameters_specs.drop(
    columns=["Values (b)", "Reference Year", "Cost Reduction to 2023 [%]"])
df_renov_parameters_specs.rename(
    columns={"Values (a)": "specs_renovation"}, inplace=True)
dict_renov_parameters_specs = df_renov_parameters_specs.to_dict()
specs_renovation = dict_renov_parameters_specs["specs_renovation"]


"_____Data_Import_____"
print("\nImporting Data ...\n")

# Creating

start = "01-01-{} 00:00".format(str(2013))
end = "31-12-{} 23:00".format(str(2013))
timestamp = pd.date_range(start=start, end=end, freq='H')  # tz='UTC'
timezone = 1  # Time Zone of system Location

# Extracting Filenames
# importing all Renovation Profiles
building_data_filenames_0 = glob.glob(
    path + "/Input/buildings/HT26 (MA Thesis)/ohne_Sanierung/*.csv")
building_data_filenames_1 = glob.glob(
    path + "/Input/buildings/HT26 (MA Thesis)/Sanierung_Dach/*.csv")
building_data_filenames_2 = glob.glob(
    path + "/Input/buildings/HT26 (MA Thesis)/Sanierung_DachFenster/*.csv")
building_data_filenames_3 = glob.glob(
    path + "/Input/buildings/HT26 (MA Thesis)/Sanierung_DachFensterWände/*.csv")
building_data_filenames_4 = glob.glob(
    path + "/Input/buildings/HT26 (MA Thesis)/Sanierung_DachFensterWändeKeller/*.csv")
building_data_filenames_5 = glob.glob(
    path + "/Input/buildings/HT26 (MA Thesis)/Sanierung_vollständig/*.csv")

atmospheric_data_filenames = glob.glob(path + "/Input/atmosphere/*")


# # Data Imports from Sheets
# Emission timeline Data Import
co2_data = pd.read_csv(
    atmospheric_data_filenames[0], sep=';', decimal=",", encoding='unicode_escape')

# Temperature Data Import
temp_data = pd.read_excel(atmospheric_data_filenames[1])

# Cooled Module Temperatures & Heat Flux Temperatures for SRTs from Research Project SolardachpfanneNRW
solar_roof_tile_data = pd.read_excel(atmospheric_data_filenames[2])

irrad_data = {"South": 3, "West": 4, "East": 5}
wind_data = {"South": 1, "West": 2, "East": 3}
wind_timeseries = pd.read_excel(
    atmospheric_data_filenames[6])  # Wind Data Import
for direction in irrad_data.keys():
    # read irradiation timeline for given direction
    irrad_data[direction] = pd.read_csv(atmospheric_data_filenames[irrad_data[direction]], sep=',',
                                        decimal=".", encoding='unicode_escape', skiprows=8, skipfooter=12, engine="python")
    irrad_data[direction]["Timestamp"] = timestamp
    irrad_data[direction]["POA Global [W/m2]"] = irrad_data[direction]["Beam inplane [W/m2]"] + \
        irrad_data[direction]["Diffuse inplane [W/m2]"]
    # Wind Speed Data Manipulation
    """
    Wind Speeds for Winds that come from a directions from "behind" the roof are set to zero.
    This should imitate a measurement of wind speeds in front of the solar power plant.
    For a south orientation of the solar power plant, all wind speeds that come from WbN to EbN are set to zero
    """
    wind_data[direction] = wind_timeseries.copy()
    if direction == "South":
        wind_data[direction].loc[~wind_data[direction]["Wind Direction [°]"].between(
            90, 270, inclusive='both'), 'Wind Speed [m/s]'] = 0  # 180 ist Süd, 0 ist Nord, 90 ist Ost und 270 ist West
    elif direction == "West":
        wind_data[direction].loc[~wind_data[direction]["Wind Direction [°]"].between(
            180, 360, inclusive='both'), 'Wind Speed [m/s]'] = 0
    elif direction == "East":
        wind_data[direction].loc[~wind_data[direction]["Wind Direction [°]"].between(
            0, 180, inclusive='both'), 'Wind Speed [m/s]'] = 0
    else:
        pass

"____Data_Handling____"
print("\nData Handling ..")

solar_roof_tile_data["Temperature Difference Heat Flux [°C]"] = solar_roof_tile_data["Tmcooling [°C]"] - \
    solar_roof_tile_data["T heatflux [°C]"]
# Deleting last two rows of Data Frame (Sum and Mean of Input data)
solar_roof_tile_data.drop(
    solar_roof_tile_data.index[len(solar_roof_tile_data)-2:], inplace=True)

# Temperature gain for Heatflux from Solar rooftile Heat - Estimate Coefficients from Regression
regression_coefficients, r_regression = heat_flux_temperature_difference(
    solar_roof_tile_data, irrad_upper=1100)

# CO2 Emissionfactor resampling
# Resample CO2 Emission Data to 8759 d/a
co2_data = signal.resample(co2_data, 8759)


"____Energy_System_Initiation____"
print("\nInitiating Energy System ..\n")

# Series Resistance [Ohm] by fit desoto function (pv lib) and manufacturer data
rs_series = fit_cec_sam(celltype=hybrid_solar_rooftiles["type"],
                        v_mp=hybrid_solar_rooftiles["Umpp"],
                        i_mp=hybrid_solar_rooftiles["Impp"],
                        v_oc=hybrid_solar_rooftiles["UL"],
                        i_sc=hybrid_solar_rooftiles["Ik"],
                        alpha_sc=hybrid_solar_rooftiles["Temperature_coeff._alpha"] *
                        hybrid_solar_rooftiles["Ik"],
                        beta_voc=hybrid_solar_rooftiles["Temperature_coeff._betha"] *
                        hybrid_solar_rooftiles["UL"],
                        gamma_pmp=hybrid_solar_rooftiles["Temperature_coeff._gamma"]*100,
                        cells_in_series=hybrid_solar_rooftiles["n cell"],
                        temp_ref=hybrid_solar_rooftiles["Temperature STC"])

# Asign fitted series resistance
hybrid_solar_rooftiles["Series Resistance Rs"] = rs_series[2]
hybrid_solar_rooftiles["solar_cell_area"] = round((hybrid_solar_rooftiles["Umpp"] * hybrid_solar_rooftiles["Impp"]) / (
    hybrid_solar_rooftiles["eta_el_stc"] / 100 * hybrid_solar_rooftiles["Irradiation STC"]), 4)  # estimated cell area
hybrid_solar_rooftiles["p_area"] = hybrid_solar_rooftiles["p_max"] / \
    hybrid_solar_rooftiles["solar_cell_area"]  # max power per area

max_number_of_tiles = int(
    building["area_roof"] * hybrid_solar_rooftiles["tiles_per_sqm"])
max_pv_power = (max_number_of_tiles * hybrid_solar_rooftiles["p_max"])

if optimization_mode == False and ((own_design.item(0)*1000 + own_design.item(1)*1000) > 2 * max_pv_power):
    print("\n\nThe choosen PV power for the own system design exceeds the maximum pv capacity of the underlying roof.\nPlease choose an installed PV capacity of <{} kWp.\nTerminating Programm\n".format(max_pv_power/1000))
    sys.exit()
else:
    pass


# This works, but normally the system component parameters have to be passed in the function call, and ultimately into the system design problem calls, which is oriented to the pymoo structure. An idea is to pass the component dicts by kwargs**.
def energy_system_simulation(designs):

    return_list_1 = []

    def threaded_function(j, designs, result_list):

        design = designs[j]

        start = "01-01-{} 00:00".format(str(2013))
        end = "31-12-{} 23:00".format(str(2013))
        timestamp = pd.date_range(start=start, end=end, freq='H')  # tz='UTC'

        # Data Imports from Sheets
        timestamp = pd.DataFrame({'Timestamp': timestamp})
        building_demand = timestamp

        # Getting Values from evaluation function of pymoo.Problem
        # Installed PV Power on west roof[W]
        pv_power_west = float(design.item(0)),
        # Installed PV Power on east roof[W]
        pv_power_east = float(design.item(1)),
        # Installed Battery Storage Capacity [kW]
        storage_capacity_electric = float(design.item(2)),
        # Installed Warm Water Storage Volume [L]
        storage_capacity_thermal = float(design.item(3)),
        # Heat Pump nominal Power [kW]
        hp_nominal_power = float(design.item(4)),
        # Heat Pump Bivalency Point [°C]
        hp_bivalency_point = float(design.item(5))

        if combined_heat_and_power == True:  # If CHP should not be simulated, the installed capacity is set to 0
            # CHP Installed capacity electric [kW]
            chp_specs["p_el_nom_chp"] = float(design.item(6))
            fuel = float(design.item(7))
            ev_ems = float(design.item(8))
            # House Renovation Status
            renovation_case = float(design.item(9))
        else:
            chp_specs["p_el_nom_chp"] = 0
            fuel = float(design.item(6))
            ev_ems = float(design.item(7))
            # House Renovation Status
            renovation_case = float(design.item(8))

        # Giving the variable fuel a discretized value and choose the right specs for the boiler fuel
        if fuel >= 0 and fuel <= 1:
            fuel = "NG"
            boiler_specs = boiler_specs_ng
        elif fuel > 1 and fuel <= 2:
            fuel = "H2"
            boiler_specs = boiler_specs_h2
        else:
            print("Fuel variable out of bounds. Execution stopped!")
            sys.exit()

        "__Loading EV Connection Profile__"
        if ev_ems >= 1:
            ev_df["Dataframe"] = ev_ems_connection_profile(  # plug-in state of the vehicle, showing when the car is avaible for charging
                year=specs_eco["Year of Investigation"],  # simulated year
                # time returning home from work
                arrival_time=ev_df["arrival time"],
                # time leaving home for work
                departure_time=ev_df["departure time"],
                # time leaving home for vacation
                vacation_departure_time=ev_df["vacation departure time"],
                # time returning home from vacation
                vacation_arrival_time=ev_df["vacation arrival time"],
                number_vacation_weekends=ev_df["number of vacations"])  # number of vacations per year

        # Adding an EV Connnection Prfile without an EMS
        else:
            ev_df["Dataframe"] = ev_normal_connection_profile(  # plug-in state of the vehicle, showing when the car is avaible for charging
                year=specs_eco["Year of Investigation"],  # simulated year
                # time returning home from work
                arrival_time=ev_df["arrival time"],
                # time leaving home for work
                departure_time=ev_df["departure time"],
                # time leaving home for vacation
                vacation_departure_time=ev_df["vacation departure time"],
                # time returning home from vacation
                vacation_arrival_time=ev_df["vacation arrival time"],
                number_vacation_weekends=ev_df["number of vacations"])  # number of vacations per year

        "__Importing Thermal Load Profile__"
        if renovation_case < 1:
            building_data_filenames = building_data_filenames_0
            # without energy retrofit

        elif renovation_case < 2:
            building_data_filenames = building_data_filenames_1
            # roof insulation

        elif renovation_case < 3:
            building_data_filenames = building_data_filenames_2
            # roof insulation and window upgrade

        elif renovation_case < 4:
            building_data_filenames = building_data_filenames_3
            # roof and wall insulation and window upgrade

        elif renovation_case < 5:
            building_data_filenames = building_data_filenames_4
            # roof, basemnet ceiling and wall insulation and window upgrade

        elif renovation_case <= 6:
            building_data_filenames = building_data_filenames_5
            # roof, basemnet ceiling and wall insulation and window upgrade and ventilation system alongside air sealing

        for file in building_data_filenames:  # HANDLING DER LOADPROFILE
            building_demand = pd.concat([building_demand, pd.read_csv(file, sep=';', decimal=",", usecols=[
                                        2])], axis=1)  # Concating all Data from Excel Sheets together in one dataframe

        "___Handling Thermal Load Profiles___"
        # Net Heating Demand
        building_demand.insert(loc=2, column='Sum Space Heating Net [kWh]',
                               value=building_demand["Sum Space Heating [kWh]"]/(building["eta_s"] * building["eta_d"]))  # Net Heating Demand = Space Heating Demand respecting losses due to distributon & Supply

        # Warm Water consumption in Energy Demand
        building_demand.insert(loc=5, column='Sum Warm Water [kWh]',
                               value=(building_demand["Sum Warm Water [L]"] * rho_w * cp_w * (ww_storage["t_tap_water"] - ww_storage["t_cw"]) / 3600) / building["eta_d"])  # Energy Demand for Tap Water [kWh_th] respecting losses due to distribution

        # Scaling electric energy demand on ~ 2700 kWh /household: 8100 kWh for all 3 households
        building_demand["Sum Electricity [kWh]"] = building_demand["Sum Electricity [kWh]"] * \
            scaling_factor_electric_energy_demand

        # Hybrid Solar roof tiles
        hybrid_solar_rooftiles["p_installed_west"] = pv_power_west[0]
        hybrid_solar_rooftiles["p_installed_east"] = pv_power_east[0]

        "__Handling EV Soc__"
        # setting values for Electric Vehicle Battery Capacity and SoC
        # Electrical vehicle's initial battery capacity
        e_electrical_vehicle_t_min_1 = ev_df["Battery Capacity"] * 0.5
        # initial state of charge according to battery size
        soc_ev_t_min_1 = e_electrical_vehicle_t_min_1 / \
            ev_df['Battery Capacity']

        building["area_solar_roof_tiles_west"] = hybrid_solar_rooftiles["p_installed_west"] / \
            hybrid_solar_rooftiles["p_area"]
        building["area_solar_roof_tiles_east"] = hybrid_solar_rooftiles["p_installed_east"] / \
            hybrid_solar_rooftiles["p_area"]
        building["area_solar_roof_tiles"] = building["area_solar_roof_tiles_east"] + \
            building["area_solar_roof_tiles_west"]

        # Calculating air volume flow(solar roof tiles area)
        fan_heat_pump["air volume flow"] = fan_heat_pump["v_ratio"] * \
            building["area_solar_roof_tiles"]
        fan_heat_pump["p_el"] = (fan_heat_pump["p_max"]/fan_heat_pump["air volume flow max"]) * \
            fan_heat_pump["air volume flow"] / \
            1000    # Interpolating p_el(air volume flow) [kW]

        # Warm Water Storage
        ww_storage["V_storage"] = storage_capacity_thermal[0]
        ww_storage["Watermass"] = rho_w * ww_storage["V_storage"]

        # Heat Exchanger
        ww_storage["solar_circle_he_area"] = ww_storage["V_storage"] * \
            ww_storage["solar_circle_he_ratio"]
        ww_storage["warm_water_he_area"] = ww_storage["V_storage"] * \
            ww_storage["warm_water_circle_he_ratio"]

        # Maximum capacity of warm water storage [kWh_th] as in DIN EN ISO 12831 (2017).  Conversions: 1h/3600s
        ww_storage["Q_storage_nom"] = ww_storage["V_storage"] * rho_w * cp_w * \
            (ww_storage["t_upper"] - ww_storage["t_lower"]) * \
            ww_storage["load_factor_f1"] * (1/3600)
        # the load factor is applied according to DIN EN 12831-3:2017. This factor is used for warm water storages with internal heat exchangers
        ww_storage["Q_storage_initial"] = ww_storage["Q_storage_nom"] * 0.8

        # heat losses of warm water storage [kW_th] as in DIN EN ISO 12831 (2017)
        q_loss_regression_coefficients, r_regression_qloss = q_loss_specific_din12831(
            combined_heat_and_power)
        ww_storage["q_storage_loss"] = q_loss_regression(
            ww_storage["V_storage"], q_loss_regression_coefficients[0][0]) * ((ww_storage["t_upper"] - ww_storage["t_a"])/45)

        # estimated minimum Temperature at the bottom of the storage to ensure the Set Temperature at the sensor
        ww_storage["t_storage_lower_min"] = abs(
            ww_storage["t_upper"]-2*ww_storage["t_sensor"])
        # q_storage_on will be the energy level the thermal storage has to contain after a timestep
        ww_storage["q_storage_on"] = ww_storage["V_storage"] * rho_w * cp_w * (1 - (ww_storage["h_sensor"]/(
            2*ww_storage["h_storage"]))) * (ww_storage["t_sensor"] - ww_storage["t_lower"]) * (1/3600) * ww_storage["load_factor_f1"]

        # HP
        # Bivalency Point
        heat_pump["Bivalency Point"] = hp_bivalency_point
        # nominal Power of Heat Pump in kW
        heat_pump["q_nom"] = hp_nominal_power[0]

        # Electrical Storage
        # Nominal Storage Capacity of electrical Storage [kWh_el]
        electrical_storage["e_electrical_storage_nom"] = storage_capacity_electric[0]
        electrical_storage["e_electrical_storage_usable"] = electrical_storage["e_electrical_storage_nom"] * \
            electrical_storage["Discharge Depth max"]  # Usable Storage Capacity of electrical Storage [kWh_el]
        # Initial Storage Capacity of electrical Storage [kWh_el]
        electrical_storage["e_electrical_storage_initial"] = electrical_storage["e_electrical_storage_usable"] * 0.5

        # the variables with t_min_1 contain the values of that variable from the timestep beforehand
        # these are set to their respective initial values
        e_ww_storage_t_min_1 = ww_storage["Q_storage_initial"]
        e_electrical_storage_t_min_1 = electrical_storage["e_electrical_storage_initial"]
        soc_t_min_1 = (electrical_storage["e_electrical_storage_initial"] /
                       electrical_storage["e_electrical_storage_usable"]) * 100

        "_____Time_Series_Simulation_____"

        energy_system_results_electrical = []
        energy_system_results_thermal = []
        warm_water_storage_temperatures = []

        # print("Simulating Population {}".format(j+1))
        for i in building_demand.index[0:8759]:

            "_____Energy_System_-_SolarThermal____"

            # Solar Rooftiles
            solar_thermal_energy = solar_rooftiles(timestep=i,
                                                   ambient_temperature=temp_data["Temperature [°C]"][i],
                                                   irradiation=irrad_data,
                                                   wind_speed=wind_data,
                                                   building=building,
                                                   hybrid_solar_rooftiles=hybrid_solar_rooftiles,
                                                   regression_coefficients=regression_coefficients
                                                   )

            "_____System_Operation_____"

            # this class determines which components are in operation. e. g. if the ambient temperature falls below the bivalency point, only the gas boiler will be working
            system_ops = energy_system_operating_strategy(irradiation=irrad_data["West"]["POA Global [W/m2]"][i] + irrad_data["East"]["POA Global [W/m2]"][i],
                                                          air_mass_flow_output_temperature=solar_thermal_energy.air_mass_flow_output_temperature,
                                                          e_ww_storage_t_min_1=e_ww_storage_t_min_1,
                                                          e_electrical_storage_t_min_1=e_electrical_storage_t_min_1,
                                                          p_pv=solar_thermal_energy.p_solar_roof_tile_electrical_ac,
                                                          building_demand=building_demand.iloc[i],
                                                          ww_storage=ww_storage,
                                                          heat_pump=heat_pump,
                                                          # Decision Variable for CHP simulation (System Configuration B)
                                                          combined_heat_and_power=combined_heat_and_power
                                                          )

            "_____Energy_System_-_Thermal____"

            # Heat Pump
            heat_pump_energies = heat_pump_ashp(t_in=solar_thermal_energy.air_mass_flow_output_temperature,
                                                heat_pump=heat_pump
                                                )

            # Warm Water Storage
            ww_storage_balance = warm_water_storage_mixed(system_ops,
                                                          heat_pump_energies,
                                                          ww_storage,
                                                          heat_pump,
                                                          cp_w,
                                                          combined_heat_and_power,
                                                          chp_specs,
                                                          e_ww_storage_t_min_1,
                                                          building_demand.iloc[i],
                                                          fan_heat_pump=fan_heat_pump
                                                          )

            # H2 Boiler
            h2_boiler_energy = h2_boiler(ww_storage_balance,
                                         system_ops,
                                         boiler_specs
                                         )

            energy_system_thermal_temp = [i, irrad_data["South"]["Timestamp"][i],
                                          ww_storage_balance.e_demand_heating_th, ww_storage_balance.e_demand_ww_th, ww_storage_balance.e_demand_th_total,
                                          heat_pump_energies.hp_cop, heat_pump_energies.q_th, h2_boiler_energy.q_boiler_th, ww_storage_balance.e_reheat_heating_circle_th,
                                          ww_storage_balance.q_th_chp, ww_storage_balance.q_th_emc, ww_storage_balance.eta_th_chp, ww_storage_balance.fe_kg_chp,
                                          ww_storage_balance.q_storage_th_in, ww_storage_balance.q_storage_th_out, ww_storage_balance.e_ww_storage_t, e_ww_storage_t_min_1,
                                          h2_boiler_energy.fe_boiler, ww_storage_balance.ops
                                          ]

            warm_water_storage_temp = [i, irrad_data["South"]["Timestamp"][i],
                                       ww_storage_balance.t_ww_storage_lower, ww_storage_balance.t_ww_storage_mean,
                                       ww_storage_balance.q_solar_circle_he, ww_storage_balance.q_warm_water_he, e_ww_storage_t_min_1
                                       ]

            "_____Energy_System_-_Electrical___"

            p_el_fan = fan_heat_pump["p_el"] * system_ops.fan

            # Positive Residual Load: energy gain or energy flow into system, Negative Load: energy demand after PV energy and system components demand (electric)
            residual_load_electric = solar_thermal_energy.p_solar_roof_tile_electrical_ac - \
                building_demand.iloc[i]["Sum Electricity [kWh]"] - h2_boiler_energy.p_boiler_el - heat_pump_energies.p_el * system_ops.hp - \
                p_el_fan + ww_storage_balance.p_el_chp - \
                ww_storage_balance.p_el_emc - ww_storage_balance.p_circulation_pumps
            # saving residuals before being stored by EV and Battery
            residual_load_electric_initial = residual_load_electric

            # Electrical Vehicle Storage
            if ev_ems >= 1:

                ev_storage_balance = electrical_vehicle_ems(residual_load_electric=residual_load_electric,
                                                            ev_df=ev_df,
                                                            ev_plug_t_min_1=ev_df["Dataframe"]["Plug"][i],
                                                            ev_plug_t_min_2=ev_df["Dataframe"]["Plug"][i+1],
                                                            e_electrical_vehicle_t_min_1=e_electrical_vehicle_t_min_1,
                                                            soc_ev_t_min_1=soc_ev_t_min_1)

            else:

                ev_storage_balance = electrical_vehicle_normal(residual_load_electric=residual_load_electric,
                                                               ev_df=ev_df,
                                                               ev_plug_t_min_1=ev_df["Dataframe"]["Plug"][i],
                                                               ev_plug_t_min_2=ev_df["Dataframe"]["Plug"][i+1],
                                                               e_electrical_vehicle_t_min_1=e_electrical_vehicle_t_min_1,
                                                               soc_ev_t_min_1=soc_ev_t_min_1)

            # update residual laods, SoC and capacity of Electrical Vehicle
            soc_ev_t_min_1 = ev_storage_balance.soc_t_min_1
            e_electrical_vehicle_t_min_1 = ev_storage_balance.e_capacity_t_min_1

            # update residual after charging car
            residual_load_electric = ev_storage_balance.residual_load_electric_2

            # Electrical Energy Storage
            electrical_storage_balance = electrical_energy_storage(residual_load_electric=residual_load_electric,
                                                                   e_electrical_storage_t_min_1=e_electrical_storage_t_min_1,
                                                                   electrical_storage=electrical_storage,
                                                                   soc_t_min_1=soc_t_min_1
                                                                   )
            # updating battery's State of Charge (SoC)
            soc_t_min_1 = electrical_storage_balance.soc
            residual_load_electric = electrical_storage_balance.residual_load_electric

            # power taken from grid, after battery discharges all its content
            p_electricity_grid = - residual_load_electric + \
                electrical_storage_balance.p_electrical_storage

            energy_system_electrical_temp = [i, ev_df["Dataframe"]["Date"][i],
                                             solar_thermal_energy.ambient_temperature, solar_thermal_energy.air_mass_flow_output_temperature, solar_thermal_energy.t_module, solar_thermal_energy.eta_solar_roof_tile,
                                             irrad_data["South"]["POA Global [W/m2]"][i], solar_thermal_energy.p_solar_roof_tile_electrical_dc, solar_thermal_energy.p_solar_roof_tile_electrical_ac, building_demand.iloc[i]["Sum Electricity [kWh]"],
                                             heat_pump_energies.p_el * system_ops.hp, ww_storage_balance.p_el_chp, ww_storage_balance.eta_el_chp, ww_storage_balance.p_el_emc, h2_boiler_energy.p_boiler_el, p_el_fan, ww_storage_balance.p_circulation_pumps,
                                             residual_load_electric, electrical_storage_balance.p_electrical_storage, electrical_storage_balance.e_electrical_storage_t, electrical_storage_balance.soc, p_electricity_grid, ev_df[
                                                 "Dataframe"]["Plug"][i],
                                             soc_ev_t_min_1, residual_load_electric_initial, ev_storage_balance.p_charge_electrical_vehicle
                                             ]

            "___Results_Timeseries_Calculation_"

            energy_system_results_electrical.append(
                energy_system_electrical_temp)
            energy_system_results_thermal.append(energy_system_thermal_temp)
            warm_water_storage_temperatures.append(warm_water_storage_temp)

            # Set Storage values of t to t-1
            e_ww_storage_t_min_1 = ww_storage_balance.e_ww_storage_t
            e_electrical_storage_t_min_1 = electrical_storage_balance.e_electrical_storage_t

        "____Creating Result Data Frames____"

        column_names_electrical = ["Index", "Timestamp", "T ambient [°C]", "T SRT Air Heatflux [°C]", "T Module [°C]", "Module Efficiency [-]", "POA Global [W/m2]", "P_el SRT dc [kW]", "P_el SRT ac [kW]", "Elecricity Demand [kWh el.]", "P_el Heat Pump [kW]", "P_el CHP [kW el.]",
                                   "Eta CHP el.", "Pel EMC [kW el.]", "P_el Boiler [kW]", "P_el Fan [kW]", "P el Circulation Pumps [kW]", "Residual Load [kW]", "P_el elec. Storage [kW]", "E elec. Storage [kWh]", "SOC [%]", "Grid [kW]", "EV Plug Status", "EV SOC [%]", "Solar Residuals", "EV P loading [kWh]"]
        energy_system_results_electrical = pd.DataFrame(
            energy_system_results_electrical, columns=column_names_electrical)

        column_names_thermal = ["Index", "Timestamp", "Heating Demand Net [kWh th.]", "Warm Water Demand [kWh th.]", "Total Thermal Energy Demand [kWh th.]", "HP COP [-]", "q HP [kW th.]", "q Boiler [kW th.]", "q Reheat Heating Circle [kW th.]", "q CHP [kW th.]",
                                "q CHP excess [kW th.]", "Eta CHP th", "H2 cons. CHP [kg H2]", "q in Warm Water Storage [kW th.]", "q out Warm Water Storage [kW th.]", "Q Warm Water Storage t [kWh th.]", "Q Warm Water Storage t-1 [kWh th.]", "Cons. Boiler [kg]", "OPS"]
        energy_system_results_thermal = pd.DataFrame(
            energy_system_results_thermal, columns=column_names_thermal)

        column_names_ww_storage = ["Index", "Timestamp", "Tu [°C]", "Tm [°C]",
                                   "Heat Exchanger SC max [kW]", "Heat Exchanger WW max [kW]", "Warm Water Storage [kWh]"]
        warm_water_storage_temperatures = pd.DataFrame(
            warm_water_storage_temperatures, columns=column_names_ww_storage)

        "____Energy_System_-_Technical_KPIs___"

        # Autarky Rate
        electricity_consumption_sum = energy_system_results_electrical["Elecricity Demand [kWh el.]"].copy() + energy_system_results_electrical["P_el Boiler [kW]"].copy() + energy_system_results_electrical["P_el Fan [kW]"].copy()\
            + energy_system_results_electrical["Pel EMC [kW el.]"].copy() + energy_system_results_electrical["P el Circulation Pumps [kW]"].copy(
        ) + energy_system_results_electrical["P_el Heat Pump [kW]"].copy() + energy_system_results_electrical["EV P loading [kWh]"].copy()  # electric consumption in energy system
        thermal_consumption_sum = energy_system_results_thermal["q Boiler [kW th.]"].copy(
        ) + energy_system_results_thermal["q CHP [kW th.]"].copy() + energy_system_results_thermal["q HP [kW th.]"].copy()  # thermal consumption in energy system
        # electricity from grid
        grid_electricity_consumption_sum = energy_system_results_electrical["Grid [kW]"].copy(
        )
        gas_consumption_sum = (energy_system_results_thermal["Cons. Boiler [kg]"].copy(
        ) + energy_system_results_thermal["H2 cons. CHP [kg H2]"].copy()) * boiler_specs["Ho"]  # gas from grid
        # autarky rate calculation regarding electrical and thermal together
        autarky_rate_system = autarky_rate_calculation(
            electricity_consumption_sum, thermal_consumption_sum, grid_electricity_consumption_sum, gas_consumption_sum)
        energy_system_results_electrical.insert(
            loc=9, column='Autarky Rate [-]', value=autarky_rate_system)

        # Heat Pump Share of Grid cover (for seperate billing via Heat Pump Tarif)

        # Copying Slices of data Frame for grid cover, Heat Pump electrical power, electrical storage output power
        p_el_heat_pump = energy_system_results_electrical["P_el Heat Pump [kW]"].copy(
        )
        p_el_storage_out = energy_system_results_electrical["P_el elec. Storage [kW]"].copy(
        )
        p_el_pv = energy_system_results_electrical["P_el SRT ac [kW]"].copy()
        p_el_chp_series = energy_system_results_electrical["P_el CHP [kW el.]"].copy(
        )
        # filter only output power of storage. -> energy that is used to power either household electricity or heat pump from storage
        p_el_storage_out[p_el_storage_out > 0] = 0

        # The share of grid cover for the heat pump is the residual load of the energy storage cover, pv energy share for the HP and CHP generation (due to tandem operation for sys. configuration B), because the heat pump has the first priority when it comes to pv power supply
        p_el_heat_pump_share_grid_cover = p_el_heat_pump - \
            p_el_pv - p_el_chp_series + p_el_storage_out
        # If the residual load is negative, a surplus even after supplying the heat pump is available
        p_el_heat_pump_share_grid_cover[p_el_heat_pump_share_grid_cover < 0] = 0
        energy_system_results_electrical.insert(loc=len(energy_system_results_electrical.columns),
                                                column='Heat Pump Grid Cover Share [kW]',
                                                value=p_el_heat_pump_share_grid_cover)

        "____Energy_System_-_Economical_KPIs___"

        components = [("PVT", building["area_solar_roof_tiles"], specs_eco["Lifetime PVT"]),             # area of solar roof tiles installed is passed, as the PVT Capex are calculated per €/m2
                      ("Fan", fan_heat_pump["p_el"],
                       specs_eco["Lifetime Fan"]),
                      ("Battery", electrical_storage["e_electrical_storage_nom"],
                       specs_eco["Lifetime Battery"]),
                      ("HP", heat_pump["q_nom"], specs_eco["Lifetime HP"]),
                      ("H2 CHP", chp_specs["p_el_nom_chp"],
                       specs_eco["Lifetime H2 CHP"]),
                      ("EMC", energy_system_results_thermal["q CHP excess [kW th.]"].max(
                      ), specs_eco["Lifetime EMC"]),
                      ("Water Storage", ww_storage["V_storage"],
                       specs_eco["Lifetime Water Storage"]),
                      ("EMS", ev_ems, specs_eco["Lifetime EMS"]),
                      ("Renovation", renovation_case, specs_eco["Period"])
                      ]

        # Determine the right Lifetime
        if fuel == "NG":
            components.append(("NG Boiler", energy_system_results_thermal["q Boiler [kW th.]"].max(
            ), specs_eco["Lifetime NG Boiler"]))
        else:
            components.append(("H2 Boiler", energy_system_results_thermal["q Boiler [kW th.]"].max(
            ), specs_eco["Lifetime H2 Boiler"]))

        components = pd.DataFrame(data=components, columns=[
                                  "component", "Capacity", "Lifetime"])
        components, electric_energy_costs, price_dynamics = overall_costs(
            components, energy_system_results_electrical, energy_system_results_thermal, specs_eco, fuel, specs_renovation, building)

        overall_system_costs = components["CAPEX [€]"].sum() + components["OPEX fix [€]"].sum() + components["OPEX var [€]"].sum(
        ) - electric_energy_costs.loc['Bilancy']["Costs/Revenue over runtime [€]"]  # electric energy costs have to be subtracted due to negative sign

        overall_system_costs_per_annum = overall_system_costs / n

        # Last row of components Data Frame should show the sum of costs
        components.loc['Sum'] = components.sum()

        "____Energy_System_-_co2_Emissions___"

        system_co2_emission_timeline, co2_emission_timelines = co2_emission_timeline(
            energy_system_results_electrical["Grid [kW]"], co2_data, year, n)  # Estimate CO2 Emissions over System Lifetime
        overall_system_co2_emissions = system_co2_emission_timeline.sum()
        if fuel == "NG":  # add emissions from ng boiler
            overall_system_co2_emissions += energy_system_results_thermal["Cons. Boiler [kg]"].sum(
            ) * specs_ecological["ng_emission_factor"] * n
        overall_system_co2_emissions_per_annum = overall_system_co2_emissions / n

        "____Wrapping_Results____"

        res_design = np.array(
            [overall_system_costs_per_annum, overall_system_co2_emissions_per_annum])
        result_designs = res_design
        result_designs = result_designs.reshape((1, 2))

        if optimization_mode == True and result_designs.shape[0] > 1:
            result_designs = np.append(result_designs, [res_design], axis=0)

        return_list = [result_designs, energy_system_results_electrical, energy_system_results_thermal, warm_water_storage_temperatures, components, electric_energy_costs, price_dynamics, system_ops, solar_thermal_energy,
                       ww_storage_balance, heat_pump_energies, h2_boiler_energy, electrical_storage_balance, overall_system_co2_emissions, overall_system_costs, system_co2_emission_timeline, co2_emission_timelines]

        return return_list

    return_list_1 = Parallel(n_jobs=optimization_specs["Parallel jobs"])(delayed(threaded_function)(
        ii, designs, return_list_1) for ii in tqdm(range(len(designs)), desc='Simulating populations..'))
    return_list_2 = []

    for i in range(len(return_list_1)):
        return_list_2.append(return_list_1[i][0])

    # Function Output in Non-Optimization Mode with CHp
    if (np.shape(designs) == (1, 9)) or (np.shape(designs) == (1, 10)):
        print("\nReturn Non-Optimization Mode results")
        # returns only first row with all results of first simulation
        return return_list_1[0]
    else:  # Function Output in Optimization Mode
        # returns only list of all results_desigs (costs and co2 emissions) in the generation being simulated
        return return_list_2


if optimization_mode == True:  # Optimization
    optimization_results, energy_system_results_electrical, energy_system_results_thermal, hyper_volume_indicator, spacing = optimize_multi_objective(optimization_mode=optimization_mode,
                                                                                                                                                      optimization_specs=optimization_specs,
                                                                                                                                                      combined_heat_and_power=combined_heat_and_power,
                                                                                                                                                      max_pv_power=max_pv_power,
                                                                                                                                                      temp_data=temp_data,
                                                                                                                                                      energy_system_simulation=energy_system_simulation,
                                                                                                                                                      own_design=own_design,
                                                                                                                                                      n=n)

else:                       # No Optimization
    optimization_results, energy_system_results_electrical, energy_system_results_thermal, hyper_volume_indicator, spacing, components = optimize_multi_objective(optimization_mode=optimization_mode,
                                                                                                                                                                  optimization_specs=optimization_specs,
                                                                                                                                                                  combined_heat_and_power=combined_heat_and_power,
                                                                                                                                                                  max_pv_power=max_pv_power,
                                                                                                                                                                  temp_data=temp_data,
                                                                                                                                                                  energy_system_simulation=energy_system_simulation,
                                                                                                                                                                  own_design=own_design,
                                                                                                                                                                  n=n)
# %% Saving Reference optimization case

opt_ref = pd.DataFrame()
opt_ref["CO2 emissions"] = optimization_results.F.T[1]
opt_ref["Annuity Cost"] = optimization_results.F.T[0]
opt_ref["ng price"] = specs_eco["ng price"]
opt_ref["h2 price"] = specs_eco["h2 price"]
opt_ref["electricity price"] = specs_eco["electricity price"]
opt_ref["hyper volume"] = hyper_volume_indicator
opt_ref["spacing"] = spacing


# %% Results_to_Excel
"____Results_to_Excel___"

if combined_heat_and_power == True:
    sys_config = "B"
else:
    sys_config = "A"

if output_results_as_table == True:
    if optimization_mode == True:

        # Print complete results to excel
        if combined_heat_and_power == True:
            # here add EV optimal results (with or without?)
            # Optimization Results Data Frame
            pareto_front_df = pd.DataFrame(optimization_results.F, columns=[
                                           "Overall Costs [€/a]", "CO2äq [kg/a]"])
            optimal_solutions_df = pd.DataFrame(optimization_results.X, columns=[
                                                "PVT East [kW el]", "PVT West [kW el]", "BESS [kWh el]", "Storage [L]", "ASHP [kW th]", "t biv [°C]", "CHP [kWel]", "Type of Boiler Fuel", "EV-EMS Case", "Renovation Case"])

        else:

            pareto_front_df = pd.DataFrame(optimization_results.F, columns=[
                                           "Overall Costs [€/a]", "CO2äq [kg/a]"])
            optimal_solutions_df = pd.DataFrame(optimization_results.X, columns=[
                                                "PVT East [kW el]", "PVT West [kW el]", "BESS [kWh el]", "Storage [L]", "ASHP [kW th]", "t biv [°C]", "Type of Boiler Fuel", "EV-EMS Case", "Renovation Case"])

        hyper_volume_spacing_df = pd.DataFrame(
            {"HV": [hyper_volume_indicator], "S": [spacing]})

        try:
            # create the results folder first if it does not exist
            folder_name = os.path.join(
                path+"\Output\Optimizations", dt.date.today().strftime('%Y-%m-%d'))
            os.makedirs(folder_name)

            pareto_front_df.to_excel(os.path.join("Output\Optimizations\{}", r'ParetoFront_Sys_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx').format(dt.date.today(
            ), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))
            optimal_solutions_df.to_excel(os.path.join("Output\Optimizations\{}", r'ParetoOptimalSolutions_Sys_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx').format(dt.date.today(
            ), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))
            hyper_volume_spacing_df.to_excel(os.path.join("Output\Optimizations\{}", r'HV_S_Sys_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx').format(dt.date.today(
            ), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))
        except:
            pareto_front_df.to_excel(os.path.join("Output\Optimizations\{}", r'ParetoFront_Sys_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx').format(dt.date.today(
            ), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))
            optimal_solutions_df.to_excel(os.path.join("Output\Optimizations\{}", r'ParetoOptimalSolutions_Sys_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx').format(dt.date.today(
            ), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))
            hyper_volume_spacing_df.to_excel(os.path.join("Output\Optimizations\{}", r'HV_S_Sys_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx').format(dt.date.today(
            ), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))

    else:
        # print non-optimization results as table
        pass
# %% Plotting Pareto Fronts for optimization case
"____Plotting_____"

warnings.filterwarnings("ignore")  # Supress Plotting Warnings

if plot_results == True:
    if optimization_mode == True:
        try:
            plot_Pareto_Fronts(optimization_mode=optimization_mode, opt_results=optimization_results.F.T,
                               optimal_solutions_df=optimal_solutions_df, sys_config=sys_config, optimization_specs=optimization_specs)

        except:
            """ Use this part for plotting imported excel results """
            # importing pareto front
            pareto_front_import = pd.read_excel(
                r"C:\Users\Fares\Documents\GitHub\Multi-Objektive-Optimization\multi-objektive-optimization\Output\Optimizations\2024-10-23\ParetoFront_Sys_A_Algorithm_NSGA2_Gen_60_Pop_50_Price_9.28_Date_2024-10-23.xlsx")
            # importing optimal solutions
            optimal_solutions_import = pd.read_excel(
                r"C:\Users\Fares\Documents\GitHub\Multi-Objektive-Optimization\multi-objektive-optimization\Output\Optimizations\2024-10-23\ParetoOptimalSolutions_Sys_A_Algorithm_NSGA2_Gen_60_Pop_50_Price_9.28_Date_2024-10-23.xlsx")

            # setting reference values
            opt_ref = pd.DataFrame()
            opt_ref["CO2 emissions"] = pareto_front_import["CO2äq [kg/a]"]
            opt_ref["Annuity Cost"] = pareto_front_import["Overall Costs [€/a]"]
            opt_ref["ng price"] = specs_eco["ng price"]
            opt_ref["h2 price"] = specs_eco["h2 price"]
            opt_ref["electricity price"] = specs_eco["electricity price"]

            # plotting ParetoFront from imported Excel
            plot_Pareto_Fronts(optimization_mode=optimization_mode, opt_results=opt_ref,
                               optimal_solutions_df=optimal_solutions_import, sys_config="A", optimization_specs=optimization_specs)

# %% Plots for none Optimization case

if plot_results == True:
    if optimization_mode == False:  # plot normal Timerseries calculation results

        # Create Array for x ticks
        # Get year of input data
        input_data_year = energy_system_results_electrical.Timestamp.dt.year[0]
        # empty array for 12 ticks for each month of the year
        n_ticks = np.zeros(12)

        for i in range(len(n_ticks)-1):
            # Get number of days for each month of year of input data
            num_days_month = calendar.monthrange(input_data_year, i+1)[1]
            # Add number of hours for each month to get xticks position in hours
            n_ticks[i+1] = n_ticks[i] + num_days_month*24

        if combined_heat_and_power == False:    # plot results without CHP operation

            # Plot settings
            plt.rcParams['mathtext.default'] = 'regular'
            plt.rcParams['font.serif'] = ' DejaVu Sans'
            plt.rcParams.update({'font.size': 14})

            # Energy System - Results electrical
            fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 12))
            energy_system_results_electrical.drop(columns=['Index',
                                                           'Timestamp',
                                                           'T ambient [°C]',
                                                           'T SRT Air Heatflux [°C]',
                                                           'T Module [°C]',
                                                           'Module Efficiency [-]',
                                                           'P_el SRT dc [kW]',
                                                           'Residual Load [kW]',
                                                           'P_el Fan [kW]',
                                                           'P el Circulation Pumps [kW]',
                                                           'P_el CHP [kW el.]',
                                                           'Eta CHP el.',
                                                           'Pel EMC [kW el.]',
                                                           'Autarky Rate [-]',
                                                           'E elec. Storage [kWh]',
                                                           'Heat Pump Grid Cover Share [kW]',
                                                           'EV SOC [%]',
                                                           "EV Plug Status",
                                                           "Solar Residuals"]
                                                  ).plot(subplots=True,
                                                         ax=axes,
                                                         xlabel='{} - {}'.format(energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[
                                                                                 0], energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[len(energy_system_results_electrical)-1]),
                                                         ylabel='Power [kW$_{el}$]',
                                                         sharey=False,
                                                         sharex=True,
                                                         title='Timeseries Siumlation Results of the Energy System Sys. A (electrical)')
            fig.tight_layout()

            for i, ax in enumerate(fig.axes):
                ax.set_xticks(ticks=n_ticks)
                ax.set_xticklabels((energy_system_results_electrical.Timestamp[n_ticks].dt.strftime(
                    '%B')), rotation=30, fontdict={'fontsize': 12, 'horizontalalignment': 'right'})

            axes[0, 0].set_ylabel("POA Global [W/m2]")
            axes[1, 0].set_ylabel("Electricity Demand [kWh$_{el}$]")
            axes[3, 0].set_ylabel("Battery SOC [%]")

            plt.savefig(path + '\\Output\\Timeseries Results\\Timeseries Energy System Sys. A Electrical {}.png'.format(
                dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")), dpi=300, format='png', bbox_inches='tight')

            # Energy System - Results thermal

            fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 12))
            energy_system_results_thermal.drop(columns=['Index',
                                                        'Timestamp',
                                                        'q in Warm Water Storage [kW th.]',
                                                        'q out Warm Water Storage [kW th.]',
                                                        'q Reheat Heating Circle [kW th.]',
                                                        'q CHP [kW th.]',
                                                        'q CHP excess [kW th.]',
                                                        'H2 cons. CHP [kg H2]',
                                                        'Eta CHP th',
                                                        'Q Warm Water Storage t-1 [kWh th.]']
                                               ).plot(subplots=True,
                                                      ax=axes,
                                                      xlabel='{} - {}'.format(energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[
                                                                              0], energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[len(energy_system_results_electrical)-1]),
                                                      ylabel='Thermal Output [kW$_{th}$]',
                                                      sharey=False,
                                                      sharex=True,
                                                      title='Timeseries Siumlation Results of the Energy System Sys. A (Thermal)')

            fig.tight_layout()

            for i, ax in enumerate(fig.axes):
                ax.set_xticks(ticks=n_ticks)
                ax.set_xticklabels((energy_system_results_electrical.Timestamp[n_ticks].dt.strftime(
                    '%B')), rotation=30, fontdict={'fontsize': 12, 'horizontalalignment': 'right'})

            axes[0, 0].set_ylabel("Space Heating [kWh$_{th}$]")
            axes[0, 1].set_ylabel("DHW Demand  [kWh$_{th}$]")
            axes[1, 0].set_ylabel("Total Demand [kWh$_{th}$]")
            axes[1, 1].set_ylabel("Heat Pump COP [-]")
            axes[3, 0].set_ylabel("WWS level [kWh$_{th}$]")
            axes[3, 1].set_ylabel("H2 Consumption [kg$_{H2}$]")

            plt.savefig(path + '\\Output\\Timeseries Results\\Timeseries Energy System Sys. A Thermal {}.png'.format(
                dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")), dpi=300, format='png', bbox_inches='tight')

            # Energy System - Electrical Vehicle
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 12))
            energy_system_results_electrical.drop(columns=["Index",
                                                           "Timestamp",
                                                           "T ambient [°C]",
                                                           "T SRT Air Heatflux [°C]",
                                                           "T Module [°C]",
                                                           "Module Efficiency [-]",
                                                           "POA Global [W/m2]",
                                                           "P_el SRT dc [kW]",
                                                           "P_el SRT ac [kW]",
                                                           "Elecricity Demand [kWh el.]",
                                                           "P_el Heat Pump [kW]", "P_el CHP [kW el.]",
                                                           "Eta CHP el.",
                                                           "Pel EMC [kW el.]",
                                                           "P_el Boiler [kW]",
                                                           "P_el Fan [kW]",
                                                           "P el Circulation Pumps [kW]",
                                                           "Residual Load [kW]",
                                                           "P_el elec. Storage [kW]",
                                                           "E elec. Storage [kWh]",
                                                           "SOC [%]",
                                                           "Grid [kW]",
                                                           'Autarky Rate [-]',
                                                           'Heat Pump Grid Cover Share [kW]']
                                                  ).plot(subplots=True,
                                                         ax=axes,
                                                         xlabel='{} - {}'.format(energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[
                                                                                 0], energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[len(energy_system_results_electrical)-1]),
                                                         ylabel='Plug-in Status',
                                                         sharey=False,
                                                         sharex=True,
                                                         title='Timeseries Siumlation Results of the Energy System Sys. A (Electrical Vehicle)')

            fig.tight_layout()

            # for i, ax in enumerate(fig.axes):
            #     ax.set_xticks(ticks=n_ticks)
            #     ax.set_xticklabels((energy_system_results_electrical.Timestamp[n_ticks].dt.strftime('%B')), rotation=30 , fontdict={'fontsize': 12, 'horizontalalignment': 'right'})

            axes[1].set_ylabel('EV SoC [%]')
            axes[2].set_ylabel('Solar Residuals [kW]')

            plt.savefig(path + '\\Output\\Timeseries Results\\Timeseries Energy System Sys. A EV {}.png'.format(
                dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")), dpi=300, format='png', bbox_inches='tight')

        elif combined_heat_and_power == True:    # plot results with CHP operation

            # Plot settings
            plt.rcParams['mathtext.default'] = 'regular'
            plt.rcParams['font.serif'] = ' DejaVu Sans'
            plt.rcParams.update({'font.size': 12})

        # Energy System - Results electrical
            fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 12))
            energy_system_results_electrical.drop(columns=['Index',
                                                           'Timestamp',
                                                           'T ambient [°C]',
                                                           'T SRT Air Heatflux [°C]',
                                                           'T Module [°C]',
                                                           'Module Efficiency [-]',
                                                           'P_el SRT dc [kW]',
                                                           'Residual Load [kW]',
                                                           'P_el Fan [kW]',
                                                           'P el Circulation Pumps [kW]',
                                                           'Pel EMC [kW el.]',
                                                           'Autarky Rate [-]',
                                                           'E elec. Storage [kWh]',
                                                           'Heat Pump Grid Cover Share [kW]',
                                                           'EV SOC [%]',
                                                           "EV Plug Status",
                                                           "Solar Residuals"]
                                                  ).plot(subplots=True,
                                                         ax=axes,
                                                         xlabel='{} - {}'.format(energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[
                                                                                 0], energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[len(energy_system_results_electrical)-1]),
                                                         ylabel='Power [kW$_{el}$]',
                                                         sharey=False,
                                                         sharex=True,
                                                         title='Timeseries Siumlation Results of the Energy System Sys. B (electrical)')  # "E elec. Storage [kWh]" SOC [%]
            fig.tight_layout()

            for i, ax in enumerate(fig.axes):
                ax.set_xticks(ticks=n_ticks)
                ax.set_xticklabels((energy_system_results_electrical.Timestamp[n_ticks].dt.strftime(
                    '%B')), rotation=30, fontdict={'fontsize': 12, 'horizontalalignment': 'right'})

            axes[0, 0].set_ylabel("POA Global [W/m2]")
            axes[1, 0].set_ylabel("Electricity Demand [kWh$_{el}$]")
            axes[2, 1].set_ylabel(r"$\eta$ CHP$_{el}$ [-]")
            axes[4, 0].set_ylabel("Battery SOC [%]")

            plt.savefig(path + '\\Output\\Timeseries Results\\Timeseries Energy System Sys. B Electrical {}.png'.format(
                dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")), dpi=300, format='png', bbox_inches='tight')

            # Energy System - Results thermal

            fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 12))
            energy_system_results_thermal.drop(columns=['Index',
                                                        'Timestamp',
                                                        'q in Warm Water Storage [kW th.]',
                                                        'q out Warm Water Storage [kW th.]',
                                                        'q Reheat Heating Circle [kW th.]',
                                                        'Q Warm Water Storage t-1 [kWh th.]',
                                                        'q CHP excess [kW th.]',
                                                        'Eta CHP th']
                                               ).plot(subplots=True,
                                                      ax=axes,
                                                      xlabel='{} - {}'.format(energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[
                                                                              0], energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[len(energy_system_results_electrical)-1]),
                                                      ylabel='Thermal Output [kW$_{th}$]',
                                                      sharey=False,
                                                      sharex=True,
                                                      title='Timeseries Siumlation Results of the Energy System Sys. B (Thermal)')

            fig.tight_layout()

            for i, ax in enumerate(fig.axes):
                ax.set_xticks(ticks=n_ticks)
                ax.set_xticklabels((energy_system_results_electrical.Timestamp[n_ticks].dt.strftime(
                    '%B')), rotation=30, fontdict={'fontsize': 12, 'horizontalalignment': 'right'})

            axes[0, 0].set_ylabel("Space Heating [kWh$_{th}$]")
            axes[0, 1].set_ylabel("DHW Demand  [kWh$_{th}$]")
            axes[1, 0].set_ylabel("Total Demand [kWh$_{th}$]")
            axes[1, 1].set_ylabel("Heat Pump COP [-]")
            axes[3, 1].set_ylabel("H2 Consumption [kg$_{H2}$]")
            axes[4, 0].set_ylabel("WWS level [kWh$_{th}$]")
            axes[4, 1].set_ylabel("H2 Consumption [kg$_{H2}$]")

            plt.savefig(path + '\\Output\\Timeseries Results\\Timeseries Energy System Sys. B Thermal {}.png'.format(
                dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")), dpi=300, format='png', bbox_inches='tight')

            # Energy System - Electrical Vehicle
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 12))
            energy_system_results_electrical.drop(columns=["Index",
                                                           "Timestamp",
                                                           "T ambient [°C]",
                                                           "T SRT Air Heatflux [°C]",
                                                           "T Module [°C]",
                                                           "Module Efficiency [-]",
                                                           "POA Global [W/m2]",
                                                           "P_el SRT dc [kW]",
                                                           "P_el SRT ac [kW]",
                                                           "Elecricity Demand [kWh el.]",
                                                           "P_el Heat Pump [kW]", "P_el CHP [kW el.]",
                                                           "Eta CHP el.",
                                                           "Pel EMC [kW el.]",
                                                           "P_el Boiler [kW]",
                                                           "P_el Fan [kW]",
                                                           "P el Circulation Pumps [kW]",
                                                           "Residual Load [kW]",
                                                           "P_el elec. Storage [kW]",
                                                           "E elec. Storage [kWh]",
                                                           "SOC [%]",
                                                           "Grid [kW]",
                                                           'Autarky Rate [-]',
                                                           'Heat Pump Grid Cover Share [kW]']
                                                  ).plot(subplots=True,
                                                         ax=axes,
                                                         xlabel='{} - {}'.format(energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[
                                                                                 0], energy_system_results_electrical.Timestamp.dt.strftime('%d.%m.%Y')[len(energy_system_results_electrical)-1]),
                                                         ylabel='Plug-in Status',
                                                         sharey=False,
                                                         sharex=True,
                                                         title='Timeseries Siumlation Results of the Energy System Sys. B (Electrical Vehicle)')

            fig.tight_layout()

            # for i, ax in enumerate(fig.axes):
            #     ax.set_xticks(ticks=n_ticks)
            #     ax.set_xticklabels((energy_system_results_electrical.Timestamp[n_ticks].dt.strftime('%B')), rotation=30 , fontdict={'fontsize': 12, 'horizontalalignment': 'right'})

            axes[1].set_ylabel('EV SoC [%]')
            axes[2].set_ylabel('Solar Residuals [kW]')

            plt.savefig(path + '\\Output\\Timeseries Results\\Timeseries Energy System Sys. B EV {}.png'.format(
                dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")), dpi=300, format='png', bbox_inches='tight')

        else:
            pass

# %% sensevitaetsanalyse


if Sensitivity_analysis == True:

    "_____Optimization_with_Sensitivity_analysis_____"
    print("Simulating NG Price Sensitivity Analysis...")
    Sensitivity_ng = sensivity_analysis(lower_boundary=0.92, Sensitivity_step=0.4, upper_boundary=3.32, sensitivity_variable="ng price",
                                        specs_eco=specs_eco,
                                        optimization_mode=optimization_mode,
                                        optimization_specs=optimization_specs,
                                        combined_heat_and_power=combined_heat_and_power,
                                        max_pv_power=max_pv_power,
                                        temp_data=temp_data,
                                        energy_system_simulation=energy_system_simulation,
                                        own_design=own_design,
                                        n=n)

    print("Simulating Electricity Price Sensitivity Analysis...")

    Sensitivity_electricity = sensivity_analysis(lower_boundary=0.22, Sensitivity_step=0.05, upper_boundary=0.47, sensitivity_variable="electricity price",
                                                 specs_eco=specs_eco,
                                                 optimization_mode=optimization_mode,
                                                 optimization_specs=optimization_specs,
                                                 combined_heat_and_power=combined_heat_and_power,
                                                 max_pv_power=max_pv_power,
                                                 temp_data=temp_data,
                                                 energy_system_simulation=energy_system_simulation,
                                                 own_design=own_design,
                                                 n=n)

    print("Simulating H2 Price Sensitivity Analysis...")
    Sensitivity_h2 = sensivity_analysis(lower_boundary=2.28, Sensitivity_step=2, upper_boundary=12.28, sensitivity_variable="h2 price",
                                        specs_eco=specs_eco,
                                        optimization_mode=optimization_mode,
                                        optimization_specs=optimization_specs,
                                        combined_heat_and_power=combined_heat_and_power,
                                        max_pv_power=max_pv_power,
                                        temp_data=temp_data,
                                        energy_system_simulation=energy_system_simulation,
                                        own_design=own_design,
                                        n=n)


# %% Sensitivity Analysis to Excel
if Sensitivity_analysis == True:
    Sensitivity_electricity.to_excel(os.path.join("Output\Optimizations\{}", r"Sensitivity_Analysis_of_Electricity_Prices_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx").format(
        dt.date.today(), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))
    Sensitivity_h2.to_excel(os.path.join("Output\Optimizations\{}", r"Sensitivity_Analysis_of_h2_Prices_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx").format(dt.date.today(
    ), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))
    Sensitivity_ng.to_excel(os.path.join("Output\Optimizations\{}", r"Sensitivity_Analysis_of_ng_Prices_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx").format(dt.date.today(
    ), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))

else:
    pass


# %% plotting Sensitivity plots

if plot_results == True:
    if Sensitivity_analysis == True:

        try:

            plotting_Sensitivity(
                opt_ref, Sensitivity_electricity, "electricity")
            plotting_Sensitivity(opt_ref, Sensitivity_h2, "h2")
            plotting_Sensitivity(opt_ref, Sensitivity_ng, "ng")

        except:
            'Enter Here the file path manually to plot results of desired files'

            # importing results from Excel tables
            # MAKE SURE TO ENTER THE CORRECT FILE PATH of Results

            # importing sensitivity analysis from excel sheets
            Sensitivity_electricity = pd.read_excel(
                r"C:\Users\Fares\Documents\GitHub\Multi-Objektive-Optimization\multi-objektive-optimization\Output\Optimizations\2024-10-23\Sensitivity_Analysis_of_Electricity_Prices_A_Algorithm_NSGA2_Gen_60_Pop_50_Price_9.28_Date_2024-10-23.xlsx")
            Sensitivity_h2 = pd.read_excel(
                r"C:\Users\Fares\Documents\GitHub\Multi-Objektive-Optimization\multi-objektive-optimization\Output\Optimizations\2024-10-23\Sensitivity_Analysis_of_h2_Prices_A_Algorithm_NSGA2_Gen_60_Pop_50_Price_9.28_Date_2024-10-23.xlsx")
            Sensitivity_ng = pd.read_excel(
                r"C:\Users\Fares\Documents\GitHub\Multi-Objektive-Optimization\multi-objektive-optimization\Output\Optimizations\2024-10-23\Sensitivity_Analysis_of_ng_Prices_A_Algorithm_NSGA2_Gen_60_Pop_50_Price_9.28_Date_2024-10-23.xlsx")

            # importing reference pareto front
            pareto_front_import = pd.read_excel(
                r"C:\Users\Fares\Documents\GitHub\Multi-Objektive-Optimization\multi-objektive-optimization\Output\Optimizations\2024-10-23\ParetoFront_Sys_A_Algorithm_NSGA2_Gen_60_Pop_50_Price_9.28_Date_2024-10-23.xlsx")

            # setting reference values
            opt_ref = pd.DataFrame()
            opt_ref["CO2 emissions"] = pareto_front_import["CO2äq [kg/a]"]
            opt_ref["Annuity Cost"] = pareto_front_import["Overall Costs [€/a]"]
            opt_ref["ng price"] = specs_eco["ng price"]
            opt_ref["h2 price"] = specs_eco["h2 price"]
            opt_ref["electricity price"] = specs_eco["electricity price"]

            # creating plots
            plotting_Sensitivity(
                opt_ref, Sensitivity_results=Sensitivity_electricity, Sensitivity_test="electricity")
            plotting_Sensitivity(opt_ref, Sensitivity_h2, "h2")
            plotting_Sensitivity(opt_ref, Sensitivity_ng, "ng")


# %% Plotting PCP
if plot_results == True:
    try:
        plot_PCP(ResultPath=path + "/Output\Optimizations/Illustrations/{}".format(dt.date.today()),
                 ParetoFrontPath_Opt_Sol=path + "/Output/Optimizations/{}/ParetoOptimalSolutions_Sys_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx".format(dt.date.today(
                 ), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")),
                 ParetoFrontPath=path + "/Output/Optimizations/{}\ParetoFront_Sys_{}_Algorithm_{}_Gen_{}_Pop_{}_Price_{}_Date_{}.xlsx".format(dt.date.today(), sys_config, optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], specs_eco["h2 price"], dt.datetime.now().strftime("%Y-%m-%d")))
    except:
        # Enter here the file path manually to regenerate needed plots
        plot_PCP(ResultPath=r"C:\Users\Fares\Documents\GitHub\Multi-Objektive-Optimization\multi-objektive-optimization\Output\Optimizations\Illustrations\2024-10-30",
                 ParetoFrontPath_Opt_Sol=r"C:\Users\Fares\Documents\GitHub\Multi-Objektive-Optimization\multi-objektive-optimization\Output\Optimizations\2024-10-23\ParetoOptimalSolutions_Sys_A_Algorithm_NSGA2_Gen_60_Pop_50_Price_9.28_Date_2024-10-23.xlsx",
                 ParetoFrontPath=r"C:\Users\Fares\Documents\GitHub\Multi-Objektive-Optimization\multi-objektive-optimization\Output\Optimizations\2024-10-23\ParetoFront_Sys_A_Algorithm_NSGA2_Gen_60_Pop_50_Price_9.28_Date_2024-10-23.xlsx")
