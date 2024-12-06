# -*- coding: utf-8 -*-
"""
- Created within the Framework of a masterthesis @
faculty of Process Engineering, Energy and Mechanical Systems 
of the Cologne University of Applied Sciences -

-> Plotting Programm of Green_Building_Opt.py
for the plotting of all important results with the optimization result data
@author: marius bartkowski

Contact:
marius.bartkowski@magenta.de or
marius.bartkowski@smail.th-koeln.de
"""

# Pymoo
from pymoo.visualization.pcp import PCP
from pymoo.mcdm.pseudo_weights import PseudoWeights

# The usual suspects
import numpy as np
import glob
import os
import pandas as pd
import datetime as dt

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

"___Data_Import___"

# Path
path = os.getcwd()  # Get dir

# Create Result Path
result_path = path + \
    "\Output\Optimizations\Illustrations\{}".format(dt.date.today())

if not os.path.exists(result_path):  # Check if result path exist
    os.makedirs(result_path)

# Extracting Filenames
# All Files for sensitivity year of investigation
sensitivity_years_all_filenames = glob.glob(
    path + "/Output/Optimizations/Sensitivity Year of Investigation/*.xlsx")
sensitivity_electricity_price_sys_a = glob.glob(
    path + "/Output/Optimizations/Sensitivity Price electricity/Sys A/*.xlsx")  # All Files for sensitivity Sys A
sensitivity_electricity_price_sys_b = glob.glob(
    path + "/Output/Optimizations/Sensitivity Price electricity/Sys B/*.xlsx")  # All Files for sensitivity Sys B
sensitivity_hydrogen_price_sys_a = glob.glob(
    path + "/Output/Optimizations/Sensitivity Price H2/Sys A/*.xlsx")  # All Files for sensitivity Sys A
sensitivity_hydrogen_price_sys_b = glob.glob(
    path + "/Output/Optimizations/Sensitivity Price H2/Sys B/*.xlsx")  # All Files for sensitivity Sys B

sensitivity_electricity_price = sensitivity_electricity_price_sys_a + \
    sensitivity_electricity_price_sys_b  # Join lists
sensitivity_hydrogen_price = sensitivity_hydrogen_price_sys_a + \
    sensitivity_hydrogen_price_sys_b

# Data Imports from Sheets
# Pareto fronts and optimal solutions from different year of investigation
for file in sensitivity_years_all_filenames:
    # building_demand = pd.concat([building_demand, pd.read_csv(file, sep=';', decimal=",", usecols=[2])], axis=1) # Concating all Data from Excel Sheets together in one dataframe
    # print(file.rfind('Pareto'))

    # All .xlsx with Pareto in filename
    if file.rfind('Pareto') != -1:
        # All Pareto Fronts
        if file.rfind('ParetoFront') != -1:
            # For system config A
            if file.rfind('Sys_A') != -1:
                # For reference Year (2023)
                if file.rfind('2023_') != -1:
                    # print(file.rfind('2023_'))
                    sensitivity_year_reference_pf_sys_a = pd.read_excel(file)
                if file.rfind('2030_') != -1:                                    # For 2030
                    sensitivity_year_2030_pf_sys_a = pd.read_excel(file)
                if file.rfind('2045_') != -1:  # for 2045
                    sensitivity_year_2045_pf_sys_a = pd.read_excel(file)
            # For system config B
            elif file.rfind('Sys_B') != -1:
                # For reference Year (2023)
                if file.rfind('2023_') != -1:
                    sensitivity_year_reference_pf_sys_b = pd.read_excel(file)
                if file.rfind('2030_') != -1:                                    # For 2030
                    sensitivity_year_2030_pf_sys_b = pd.read_excel(file)
                if file.rfind('2045_') != -1:  # for 2045
                    sensitivity_year_2045_pf_sys_b = pd.read_excel(file)

        # All Pareto Optimal Solutions
        elif file.rfind('ParetoOptimalSolutions') != -1:
            # See above ...
            if file.rfind('Sys_A') != -1:
                if file.rfind('2023_') != -1:
                    sensitivity_year_reference_pos_sys_a = pd.read_excel(file)
                if file.rfind('2030_') != -1:
                    sensitivity_year_2030_pos_sys_a = pd.read_excel(file)
                if file.rfind('2045_') != -1:
                    sensitivity_year_2045_pos_sys_a = pd.read_excel(file)
            elif file.rfind('Sys_B') != -1:
                if file.rfind('2023_') != -1:
                    sensitivity_year_reference_pos_sys_b = pd.read_excel(file)
                if file.rfind('2030_') != -1:
                    sensitivity_year_2030_pos_sys_b = pd.read_excel(file)
                if file.rfind('2045_') != -1:
                    sensitivity_year_2045_pos_sys_b = pd.read_excel(file)
        else:
            pass  # Do Nothing
    else:
        pass  # Do Nothing

# Sensitivity analysis for electricity prices (for 2023 only)
# Investigated electricity price range
price_range = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
counter = 0  # initiate counter

for price in price_range:  # iterating through Price steps

    # create String with Price for iteration i for finding it in filenames
    price_str = "Price_{}".format(price)
    for file in sensitivity_electricity_price:  # Iterating through files

        # All .xlsx with ParetoFront in filename
        if file.rfind('ParetoFront') != -1:
            # All files for Sys. A
            if file.rfind('Sys_A') != -1:
                # File corresponding to the price
                if file.rfind(price_str) != -1:
                    # Import file as DataFrame
                    sensitivity_a_temp = pd.read_excel(file)
                    # Drop Unnamed: 0
                    sensitivity_a_temp.drop(columns="Unnamed: 0", inplace=True)
                    # Add Column with price
                    sensitivity_a_temp.insert(
                        loc=0, column='Electricity Price [€/kWh]', value=price)
            # Do the syme for Sys B
            if file.rfind('Sys_B') != -1:
                if file.rfind(price_str) != -1:
                    # Import file as DataFrame
                    sensitivity_b_temp = pd.read_excel(file)
                    # Drop Unnamed: 0
                    sensitivity_b_temp.drop(columns="Unnamed: 0", inplace=True)
                    sensitivity_b_temp.insert(
                        loc=0, column='Electricity Price [€/kWh]', value=price)
                else:
                    pass  # Do Nothing
            else:
                pass  # Do Nothing
        else:
            pass  # Do Nothing

    if counter == 0:
        sensitivity_elec_price_sys_a = sensitivity_a_temp
        sensitivity_elec_price_sys_b = sensitivity_b_temp
    else:
        sensitivity_elec_price_sys_a = sensitivity_elec_price_sys_a.append(
            sensitivity_a_temp)
        sensitivity_elec_price_sys_b = sensitivity_elec_price_sys_b.append(
            sensitivity_b_temp)

    counter += 1  # Increment counter

sensitivity_elec_price_sys_a = sensitivity_elec_price_sys_a.reset_index(
    drop=True)
sensitivity_elec_price_sys_b = sensitivity_elec_price_sys_b.reset_index(
    drop=True)


# Sensitivity analysis for Hydrogen prices (for 2023 only)
# Investigated electricity price range
price_range_h2 = [1, 2, 4, 6, 8, 10, 12, 14]
counter = 0  # initiate counter

for price in price_range_h2:  # iterating through Price steps

    # create String with Price for iteration i for finding it in filenames
    price_str = "Price_{}".format(price)
    for file in sensitivity_hydrogen_price:  # Iterating through files

        # All .xlsx with ParetoFront in filename
        if file.rfind('ParetoFront') != -1:
            # All files for Sys. A
            if file.rfind('Sys_A') != -1:
                # File corresponding to the price
                if file.rfind(price_str) != -1:
                    # Import file as DataFrame
                    sensitivity_a_temp = pd.read_excel(file)
                    # Drop Unnamed: 0
                    sensitivity_a_temp.drop(columns="Unnamed: 0", inplace=True)
                    # Add Column with price
                    sensitivity_a_temp.insert(
                        loc=0, column='Hydrogen Price [€/kg]', value=price)
            # Do the syme for Sys B
            if file.rfind('Sys_B') != -1:
                if file.rfind(price_str) != -1:
                    # Import file as DataFrame
                    sensitivity_b_temp = pd.read_excel(file)
                    # Drop Unnamed: 0
                    sensitivity_b_temp.drop(columns="Unnamed: 0", inplace=True)
                    sensitivity_b_temp.insert(
                        loc=0, column='Hydrogen Price [€/kg]', value=price)
                else:
                    pass  # Do Nothing
            else:
                pass  # Do Nothing
        else:
            pass  # Do Nothing

    if counter == 0:
        sensitivity_hydrogen_price_sys_a = sensitivity_a_temp
        sensitivity_hydrogen_price_sys_b = sensitivity_b_temp
    else:
        sensitivity_hydrogen_price_sys_a = sensitivity_hydrogen_price_sys_a.append(
            sensitivity_a_temp)
        sensitivity_hydrogen_price_sys_b = sensitivity_hydrogen_price_sys_b.append(
            sensitivity_b_temp)

    counter += 1  # Increment counter

sensitivity_hydrogen_price_sys_a = sensitivity_hydrogen_price_sys_a.reset_index(
    drop=True)
sensitivity_hydrogen_price_sys_b = sensitivity_hydrogen_price_sys_b.reset_index(
    drop=True)


"____Data_Handling____"

# MCDM Trade-Off Point for PF Solutions
# For estimating weigthed vector. Weight for objectives: 50/50
weights = np.array([0.5, 0.5])

# Create Arrays for trade-off estimation
sensitivity_year_reference_pf_sys_a_tradeoff = np.array(
    [sensitivity_year_reference_pf_sys_a.iloc[:, 1], sensitivity_year_reference_pf_sys_a.iloc[:, 2]])  # Sys. A 2023
sensitivity_year_reference_pf_sys_b_tradeoff = np.array(
    [sensitivity_year_reference_pf_sys_b.iloc[:, 1], sensitivity_year_reference_pf_sys_b.iloc[:, 2]])  # Sys. B 2023
sensitivity_year_2030_pf_sys_a_tradeoff = np.array(
    [sensitivity_year_2030_pf_sys_a.iloc[:, 1], sensitivity_year_2030_pf_sys_a.iloc[:, 2]])  # Sys. A 2030
sensitivity_year_2030_pf_sys_b_tradeoff = np.array(
    [sensitivity_year_2030_pf_sys_b.iloc[:, 1], sensitivity_year_2030_pf_sys_b.iloc[:, 2]])  # Sys. B 2030
sensitivity_year_2045_pf_sys_a_tradeoff = np.array(
    [sensitivity_year_2045_pf_sys_a.iloc[:, 1], sensitivity_year_2045_pf_sys_a.iloc[:, 2]])  # Sys. A 2045
sensitivity_year_2045_pf_sys_b_tradeoff = np.array(
    [sensitivity_year_2045_pf_sys_b.iloc[:, 1], sensitivity_year_2045_pf_sys_b.iloc[:, 2]])  # Sys. B 2045

# Transpose
sensitivity_year_reference_pf_sys_a_tradeoff = np.transpose(
    sensitivity_year_reference_pf_sys_a_tradeoff)
sensitivity_year_reference_pf_sys_b_tradeoff = np.transpose(
    sensitivity_year_reference_pf_sys_b_tradeoff)
sensitivity_year_2030_pf_sys_a_tradeoff = np.transpose(
    sensitivity_year_2030_pf_sys_a_tradeoff)
sensitivity_year_2030_pf_sys_b_tradeoff = np.transpose(
    sensitivity_year_2030_pf_sys_b_tradeoff)
sensitivity_year_2045_pf_sys_a_tradeoff = np.transpose(
    sensitivity_year_2045_pf_sys_a_tradeoff)
sensitivity_year_2045_pf_sys_b_tradeoff = np.transpose(
    sensitivity_year_2045_pf_sys_b_tradeoff)

# Estimate all trade-off points
sensitivity_year_reference_pf_sys_a_tradeoff, pseudo_weights_2023_sys_a_tradeoff = PseudoWeights(
    weights).do(sensitivity_year_reference_pf_sys_a_tradeoff, return_pseudo_weights=True)
sensitivity_year_reference_pf_sys_b_tradeoff, pseudo_weights_2023_sys_b_tradeoff = PseudoWeights(
    weights).do(sensitivity_year_reference_pf_sys_b_tradeoff, return_pseudo_weights=True)
sensitivity_year_2030_pf_sys_a_tradeoff, pseudo_weights_sys_2030_a_tradeoff = PseudoWeights(
    weights).do(sensitivity_year_2030_pf_sys_a_tradeoff, return_pseudo_weights=True)
sensitivity_year_2030_pf_sys_b_tradeoff, pseudo_weights_sys_2030_b_tradeoff = PseudoWeights(
    weights).do(sensitivity_year_2030_pf_sys_b_tradeoff, return_pseudo_weights=True)
sensitivity_year_2045_pf_sys_a_tradeoff, pseudo_weights_sys_2045_a_tradeoff = PseudoWeights(
    weights).do(sensitivity_year_2045_pf_sys_a_tradeoff, return_pseudo_weights=True)
sensitivity_year_2045_pf_sys_b_tradeoff, pseudo_weights_sys_2045_b_tradeoff = PseudoWeights(
    weights).do(sensitivity_year_2045_pf_sys_b_tradeoff, return_pseudo_weights=True)


# Data for PCP Plots
# Reference year (2023), Sys. A.
sensitivity_year_reference_pos_sys_a_pcp = np.array([sensitivity_year_reference_pos_sys_a.iloc[:, 1],
                                                     sensitivity_year_reference_pos_sys_a.iloc[:, 2],
                                                     sensitivity_year_reference_pos_sys_a.iloc[:, 3],
                                                     sensitivity_year_reference_pos_sys_a.iloc[:, 4],
                                                     sensitivity_year_reference_pos_sys_a.iloc[:, 5],
                                                     sensitivity_year_reference_pf_sys_a.iloc[:, 2],
                                                     sensitivity_year_reference_pf_sys_a.iloc[:, 1]]
                                                    )

# Reference year (2023), Sys. B.
sensitivity_year_reference_pos_sys_b_pcp = np.array([sensitivity_year_reference_pos_sys_b.iloc[:, 1],
                                                     sensitivity_year_reference_pos_sys_b.iloc[:, 2],
                                                     sensitivity_year_reference_pos_sys_b.iloc[:, 3],
                                                     sensitivity_year_reference_pos_sys_b.iloc[:, 4],
                                                     sensitivity_year_reference_pos_sys_b.iloc[:, 5],
                                                     sensitivity_year_reference_pos_sys_b.iloc[:, 6],
                                                     sensitivity_year_reference_pf_sys_b.iloc[:, 2],
                                                     sensitivity_year_reference_pf_sys_b.iloc[:, 1]]
                                                    )

# 2030, Sys. A.
sensitivity_year_2030_pos_sys_a_pcp = np.array([sensitivity_year_2030_pos_sys_a.iloc[:, 1],
                                                sensitivity_year_2030_pos_sys_a.iloc[:, 2],
                                                sensitivity_year_2030_pos_sys_a.iloc[:, 3],
                                                sensitivity_year_2030_pos_sys_a.iloc[:, 4],
                                                sensitivity_year_2030_pos_sys_a.iloc[:, 5],
                                                sensitivity_year_2030_pf_sys_a.iloc[:, 2],
                                                sensitivity_year_2030_pf_sys_a.iloc[:, 1]]
                                               )

# 2030, Sys. B.
sensitivity_year_2030_pos_sys_b_pcp = np.array([sensitivity_year_2030_pos_sys_b.iloc[:, 1],
                                                sensitivity_year_2030_pos_sys_b.iloc[:, 2],
                                                sensitivity_year_2030_pos_sys_b.iloc[:, 3],
                                                sensitivity_year_2030_pos_sys_b.iloc[:, 4],
                                                sensitivity_year_2030_pos_sys_b.iloc[:, 5],
                                                sensitivity_year_2030_pos_sys_b.iloc[:, 6],
                                                sensitivity_year_2030_pf_sys_b.iloc[:, 2],
                                                sensitivity_year_2030_pf_sys_b.iloc[:, 1]]
                                               )

# 2045, Sys. A.
sensitivity_year_2045_pos_sys_a_pcp = np.array([sensitivity_year_2045_pos_sys_a.iloc[:, 1],
                                                sensitivity_year_2045_pos_sys_a.iloc[:, 2],
                                                sensitivity_year_2045_pos_sys_a.iloc[:, 3],
                                                sensitivity_year_2045_pos_sys_a.iloc[:, 4],
                                                sensitivity_year_2045_pos_sys_a.iloc[:, 5],
                                                sensitivity_year_2045_pf_sys_a.iloc[:, 2],
                                                sensitivity_year_2045_pf_sys_a.iloc[:, 1]]
                                               )

# 2045, Sys. B.
sensitivity_year_2045_pos_sys_b_pcp = np.array([sensitivity_year_2045_pos_sys_b.iloc[:, 1],
                                                sensitivity_year_2045_pos_sys_b.iloc[:, 2],
                                                sensitivity_year_2045_pos_sys_b.iloc[:, 3],
                                                sensitivity_year_2045_pos_sys_b.iloc[:, 4],
                                                sensitivity_year_2045_pos_sys_b.iloc[:, 5],
                                                sensitivity_year_2045_pos_sys_b.iloc[:, 6],
                                                sensitivity_year_2045_pf_sys_b.iloc[:, 2],
                                                sensitivity_year_2045_pf_sys_b.iloc[:, 1]]
                                               )
# Transpose
sensitivity_year_reference_pos_sys_a_pcp = np.transpose(
    sensitivity_year_reference_pos_sys_a_pcp)
sensitivity_year_reference_pos_sys_b_pcp = np.transpose(
    sensitivity_year_reference_pos_sys_b_pcp)
sensitivity_year_2030_pos_sys_a_pcp = np.transpose(
    sensitivity_year_2030_pos_sys_a_pcp)
sensitivity_year_2030_pos_sys_b_pcp = np.transpose(
    sensitivity_year_2030_pos_sys_b_pcp)
sensitivity_year_2045_pos_sys_a_pcp = np.transpose(
    sensitivity_year_2045_pos_sys_a_pcp)
sensitivity_year_2045_pos_sys_b_pcp = np.transpose(
    sensitivity_year_2045_pos_sys_b_pcp)


"___Plotting:Sensitivity Year of Investigation___"

# Pareto Fronts
pf_text = "Algorithm: {}\nGenerations: {}\nPopulations: {}\n".format(
    "NSGA-II", 60, 50)

# 2023
fig, ax = plt.subplots(figsize=(9, 6))
plt.rcParams.update({'font.size': 11, "legend.loc": "lower left"})
plt.plot(sensitivity_year_reference_pf_sys_a["Overall Costs [€/a]"], sensitivity_year_reference_pf_sys_a["CO2äq [kg/a]"],
         '^', markersize=10, markerfacecolor='none', markeredgecolor='darkred', label='PF$^A$ - Sys. A')
plt.plot(sensitivity_year_reference_pf_sys_b["Overall Costs [€/a]"], sensitivity_year_reference_pf_sys_b["CO2äq [kg/a]"],
         '^', markersize=10, markerfacecolor='none', markeredgecolor='navy', label='PF$^A$ - Sys. B')
# Tradeoff points
plt.plot(sensitivity_year_reference_pf_sys_a["Overall Costs [€/a]"][sensitivity_year_reference_pf_sys_a_tradeoff],
         sensitivity_year_reference_pf_sys_a["CO2äq [kg/a]"][sensitivity_year_reference_pf_sys_a_tradeoff],
         '^', markersize=10, markerfacecolor='darkred', markeredgecolor='darkred', label="eq. Weigthed Trade-Off - Sys. A")
plt.plot(sensitivity_year_reference_pf_sys_b["Overall Costs [€/a]"][sensitivity_year_reference_pf_sys_b_tradeoff],
         sensitivity_year_reference_pf_sys_b["CO2äq [kg/a]"][sensitivity_year_reference_pf_sys_b_tradeoff],
         '^', markersize=10, markerfacecolor='navy', markeredgecolor='navy', label="eq. Weigthed Trade-Off - Sys. B")

plt.arrow(x=sensitivity_year_reference_pf_sys_a["Overall Costs [€/a]"].min(),
          y=sensitivity_year_reference_pf_sys_a["CO2äq [kg/a]"].min(),
          dx=(sensitivity_year_reference_pf_sys_a["Overall Costs [€/a]"][sensitivity_year_reference_pf_sys_a_tradeoff] -
              sensitivity_year_reference_pf_sys_a["Overall Costs [€/a]"].min())*1.3,
          dy=(sensitivity_year_reference_pf_sys_a["CO2äq [kg/a]"][sensitivity_year_reference_pf_sys_a_tradeoff] -
              sensitivity_year_reference_pf_sys_a["CO2äq [kg/a]"].min())*1.3,
          head_width=50,
          head_length=150,
          alpha=0.7,
          color='black'
          )

plt.arrow(x=sensitivity_year_reference_pf_sys_a["Overall Costs [€/a]"].min(),
          y=sensitivity_year_reference_pf_sys_a["CO2äq [kg/a]"].min(),
          dx=(sensitivity_year_reference_pf_sys_b["Overall Costs [€/a]"][sensitivity_year_reference_pf_sys_b_tradeoff] -
              sensitivity_year_reference_pf_sys_a["Overall Costs [€/a]"].min())*1.3,
          dy=(sensitivity_year_reference_pf_sys_b["CO2äq [kg/a]"][sensitivity_year_reference_pf_sys_b_tradeoff] -
              sensitivity_year_reference_pf_sys_a["CO2äq [kg/a]"].min())*1.3,
          head_width=50,
          head_length=150,
          alpha=0.7,
          color='black'
          )

plt.ylabel("CO$_{2,äq}$ Emissions [kg$\cdot a^{-1}$]")
plt.xlabel("Total Annuity System Costs  [€$\cdot a^{-1}$]")
plt.text(0.78, 0.83, pf_text, transform=ax.transAxes,  fontdict={
         'fontsize': 11, 'horizontalalignment': 'left', 'verticalalignment': 'bottom'})
plt.legend(fontsize=9)

plt.savefig(result_path + '\\PF_Reference_year_{}.png'.format(dt.date.today()),
            dpi=300, format='png')

# 2030 & 2045 in subplots

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].plot(sensitivity_year_2030_pf_sys_a["Overall Costs [€/a]"], sensitivity_year_2030_pf_sys_a["CO2äq [kg/a]"],
             '^', markersize=10, markerfacecolor='none', markeredgecolor='darkred', label='PF$^A$ - Sys. A')
axes[0].plot(sensitivity_year_2030_pf_sys_b["Overall Costs [€/a]"], sensitivity_year_2030_pf_sys_b["CO2äq [kg/a]"],
             '^', markersize=10, markerfacecolor='none', markeredgecolor='navy', label='PF$^A$ - Sys. B')
axes[0].plot(sensitivity_year_2030_pf_sys_a["Overall Costs [€/a]"][sensitivity_year_2030_pf_sys_a_tradeoff],
             sensitivity_year_2030_pf_sys_a["CO2äq [kg/a]"][sensitivity_year_2030_pf_sys_a_tradeoff],
             '^', markersize=10, markerfacecolor='darkred', markeredgecolor='darkred', label="Weigthed Trade-Off - Sys. A")
axes[0].plot(sensitivity_year_2030_pf_sys_b["Overall Costs [€/a]"][sensitivity_year_2030_pf_sys_b_tradeoff],
             sensitivity_year_2030_pf_sys_b["CO2äq [kg/a]"][sensitivity_year_2030_pf_sys_b_tradeoff],
             '^', markersize=10, markerfacecolor='navy', markeredgecolor='navy', label="Weigthed Trade-Off - Sys. B")

axes[1].plot(sensitivity_year_2045_pf_sys_a["Overall Costs [€/a]"], sensitivity_year_2045_pf_sys_a["CO2äq [kg/a]"],
             '^', markersize=10, markerfacecolor='none', markeredgecolor='darkred', label='PF$^A$ - Sys. A')
axes[1].plot(sensitivity_year_2045_pf_sys_b["Overall Costs [€/a]"], sensitivity_year_2045_pf_sys_b["CO2äq [kg/a]"],
             '^', markersize=10, markerfacecolor='none', markeredgecolor='navy', label='PF$^A$ - Sys. B')
axes[1].plot(sensitivity_year_2045_pf_sys_a["Overall Costs [€/a]"][sensitivity_year_2045_pf_sys_a_tradeoff],
             sensitivity_year_2045_pf_sys_a["CO2äq [kg/a]"][sensitivity_year_2045_pf_sys_a_tradeoff],
             '^', markersize=10, markerfacecolor='darkred', markeredgecolor='darkred', label="eq. Weigthed Trade-Off - Sys. A")
axes[1].plot(sensitivity_year_2045_pf_sys_b["Overall Costs [€/a]"][sensitivity_year_2045_pf_sys_b_tradeoff],
             sensitivity_year_2045_pf_sys_b["CO2äq [kg/a]"][sensitivity_year_2045_pf_sys_b_tradeoff],
             '^', markersize=10, markerfacecolor='navy', markeredgecolor='navy', label="eq. Weigthed Trade-Off - Sys. B")

axes[0].set_ylabel("CO$_{2,äq}$ Emissions [kg$\cdot a^{-1}$]")
axes[0].set_xlabel("Total Annuity System Costs  [€$\cdot a^{-1}$]")
axes[1].set_xlabel("Total Annuity System Costs  [€$\cdot a^{-1}$]")
axes[0].set_title("2030")
axes[1].set_title("2045")
axes[0].legend(fontsize=8, loc="upper right")
axes[1].legend(fontsize=8, loc="upper right")


# Parrallel coordinate plots (PCPs)
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.serif'] = ' DejaVu Sans'
plt.rcParams.update({'font.size': 13})
# plt.rcParams["legend.title_fontsize"] = 'x-small'

"""
colors = np.linspace(start=sensitivity_year_reference_pf_sys_a.iloc[:,2].min(),
                     stop=sensitivity_year_reference_pf_sys_a.iloc[:,2].max(), 
                     num=len(sensitivity_year_reference_pf_sys_a),
                     endpoint=True, retstep=False, dtype=None, axis=0)
"""

plt.savefig(result_path + '\\PF_2030_2045_{}.png'.format(dt.date.today()),
            dpi=300, format='png')

# 2023 - w/o CHP
pcp_plot_a_2023 = PCP(cmap='cividis', figsize=(15, 8), labels=["P$_{PVT}$\n[kWp$_{el}$]", "E$_{BESS}$\n[kWh$_{el}$]", "V$_{Storage}$\n[L]", "Q$_{ASHP}$\n[kW$_{th}$]",
                      "t$_{biv}$\n[°C]", "CO$_{2,äq}$\n[kg$\cdot$a$^{-1}$]", "c$_{total,annuity}$\n[kg$\cdot$a$^{-1}$]"], legend='check', tight_layout=True)
pcp_plot_a_2023.set_axis_style(color="dimgrey")  # , alpha=0.5)
pcp_plot_a_2023.add(sensitivity_year_reference_pos_sys_a_pcp,
                    alpha=0.3)  # alpha=0.3
pcp_plot_a_2023.add(sensitivity_year_reference_pos_sys_a_pcp[0], alpha=0.9,
                    color="darkred", linewidth=2.5, label="min. c$_{total,annuity}$")
pcp_plot_a_2023.add(
    sensitivity_year_reference_pos_sys_a_pcp[1], alpha=0.9, color='olivedrab', linewidth=2.5, label="min. CO$_{2,äq}$")
pcp_plot_a_2023.add(sensitivity_year_reference_pos_sys_a_pcp[sensitivity_year_reference_pf_sys_a_tradeoff],
                    linestyle='-.', alpha=0.5, color='darkcyan', linewidth=3, label="eq. w. Trade-Off")
pcp_plot_a_2023.show()

plt.savefig(result_path + '\\PCP_Sys_A_Reference_year_{}.png'.format(
    dt.date.today()), dpi=300, format='png')

# 2023 w CHP
pcp_plot_b_2023 = PCP(cmap='cividis', figsize=(15, 8), labels=["P$_{PVT}$\n[kWp$_{el}$]", "E$_{BESS}$\n[kWh$_{el}$]", "V$_{Storage}$\n[L]", "Q$_{ASHP}$\n[kW$_{th}$]",
                      "t$_{biv}$\n[°C]", "P$_{CHP}$\n[kW$_{el}$]", "CO$_{2,äq}$\n[kg$\cdot$a$^{-1}$]", "c$_{total,annuity}$\n[kg$\cdot$a$^{-1}$]"], legend='check', tight_layout=True)
pcp_plot_b_2023.set_axis_style(color="dimgrey")  # , alpha=0.5)
pcp_plot_b_2023.add(sensitivity_year_reference_pos_sys_b_pcp,
                    alpha=0.3)  # alpha=0.3
pcp_plot_b_2023.add(sensitivity_year_reference_pos_sys_b_pcp[1], alpha=0.9,
                    color="darkred", linewidth=2.5, label="min. c$_{total,annuity}$")
pcp_plot_b_2023.add(
    sensitivity_year_reference_pos_sys_b_pcp[0], alpha=0.9, color='olivedrab', linewidth=2.5, label="min. CO$_{2,äq}$")
pcp_plot_b_2023.add(sensitivity_year_reference_pos_sys_b_pcp[sensitivity_year_reference_pf_sys_b_tradeoff],
                    linestyle='-.', alpha=0.5, color='darkcyan', linewidth=3, label="eq. w. Trade-Off")
pcp_plot_b_2023.show()

plt.savefig(result_path + '\\PCP_Sys_B_Reference_year_{}.png'.format(
    dt.date.today()), dpi=300, format='png')

# 2030 - w/o CHP
pcp_plot_a_2030 = PCP(cmap='cividis', figsize=(15, 8), labels=["P$_{PVT}$\n[kWp$_{el}$]", "E$_{BESS}$\n[kWh$_{el}$]", "V$_{Storage}$\n[L]", "Q$_{ASHP}$\n[kW$_{th}$]",
                      "t$_{biv}$\n[°C]", "CO$_{2,äq}$\n[kg$\cdot$a$^{-1}$]", "c$_{total,annuity}$\n[kg$\cdot$a$^{-1}$]"], legend='check', tight_layout=True)
pcp_plot_a_2030.set_axis_style(color="dimgrey")  # , alpha=0.5)
pcp_plot_a_2030.add(sensitivity_year_2030_pos_sys_a_pcp,
                    alpha=0.3)  # alpha=0.3
pcp_plot_a_2030.add(sensitivity_year_2030_pos_sys_a_pcp[0], alpha=0.9,
                    color="darkred", linewidth=2.5, label="min. c$_{total,annuity}$")
pcp_plot_a_2030.add(sensitivity_year_2030_pos_sys_a_pcp[1], alpha=0.9,
                    color='olivedrab', linewidth=2.5, label="min. CO$_{2,äq}$")
pcp_plot_a_2030.add(sensitivity_year_2030_pos_sys_a_pcp[sensitivity_year_2030_pf_sys_a_tradeoff],
                    linestyle='-.', alpha=0.5, color='darkcyan', linewidth=3, label="eq. w. Trade-Off")
pcp_plot_a_2030.show()

plt.savefig(result_path + '\\PCP_Sys_A_2030_{}.png'.format(dt.date.today()),
            dpi=300, format='png')

# 2030 w/ CHP
pcp_plot_b_2030 = PCP(cmap='cividis', figsize=(15, 8), labels=["P$_{PVT}$\n[kWp$_{el}$]", "E$_{BESS}$\n[kWh$_{el}$]", "V$_{Storage}$\n[L]", "Q$_{ASHP}$\n[kW$_{th}$]",
                      "t$_{biv}$\n[°C]", "P$_{CHP}$\n[kW$_{el}$]", "CO$_{2,äq}$\n[kg$\cdot$a$^{-1}$]", "c$_{total,annuity}$\n[kg$\cdot$a$^{-1}$]"], legend='check', tight_layout=True)
pcp_plot_b_2030.set_axis_style(color="dimgrey")  # , alpha=0.5)
pcp_plot_b_2030.add(sensitivity_year_2030_pos_sys_b_pcp,
                    alpha=0.3)  # alpha=0.3
pcp_plot_b_2030.add(sensitivity_year_2030_pos_sys_b_pcp[0], alpha=0.9,
                    color="darkred", linewidth=2.5, label="min. c$_{total,annuity}$")
pcp_plot_b_2030.add(sensitivity_year_2030_pos_sys_b_pcp[1], alpha=0.9,
                    color='olivedrab', linewidth=2.5, label="min. CO$_{2,äq}$")
pcp_plot_b_2030.add(sensitivity_year_2030_pos_sys_b_pcp[sensitivity_year_2030_pf_sys_b_tradeoff],
                    linestyle='-.', alpha=0.5, color='darkcyan', linewidth=3, label="eq. w. Trade-Off")
pcp_plot_b_2030.show()

plt.savefig(result_path + '\\PCP_Sys_B_2030_{}.png'.format(dt.date.today()),
            dpi=300, format='png')

# 2045 - w/o CHP
pcp_plot_a_2045 = PCP(cmap='cividis', figsize=(15, 8), labels=["P$_{PVT}$\n[kWp$_{el}$]", "E$_{BESS}$\n[kWh$_{el}$]", "V$_{Storage}$\n[L]", "Q$_{ASHP}$\n[kW$_{th}$]",
                      "t$_{biv}$\n[°C]", "CO$_{2,äq}$\n[kg$\cdot$a$^{-1}$]", "c$_{total,annuity}$\n[kg$\cdot$a$^{-1}$]"], legend='check', tight_layout=True)
pcp_plot_a_2045.set_axis_style(color="dimgrey")  # , alpha=0.5)
pcp_plot_a_2045.add(sensitivity_year_2045_pos_sys_a_pcp,
                    alpha=0.3)  # alpha=0.3
pcp_plot_a_2045.add(sensitivity_year_2045_pos_sys_a_pcp[1], alpha=0.9,
                    color="darkred", linewidth=2.5, label="min. c$_{total,annuity}$")
pcp_plot_a_2045.add(sensitivity_year_2045_pos_sys_a_pcp[0], alpha=0.9,
                    color='olivedrab', linewidth=2.5, label="min. CO$_{2,äq}$")
pcp_plot_a_2045.add(sensitivity_year_2045_pos_sys_a_pcp[sensitivity_year_2045_pf_sys_a_tradeoff],
                    linestyle='-.', alpha=0.5, color='darkcyan', linewidth=3, label="eq. w. Trade-Off")
pcp_plot_a_2045.show()

plt.savefig(result_path + '\\PCP_Sys_A_2045_{}.png'.format(dt.date.today()),
            dpi=300, format='png')

# 2045 w/ CHP
pcp_plot_b_2045 = PCP(cmap='cividis', figsize=(15, 8), labels=["P$_{PVT}$\n[kWp$_{el}$]", "E$_{BESS}$\n[kWh$_{el}$]", "V$_{Storage}$\n[L]", "Q$_{ASHP}$\n[kW$_{th}$]",
                      "t$_{biv}$\n[°C]", "P$_{CHP}$\n[kW$_{el}$]", "CO$_{2,äq}$\n[kg$\cdot$a$^{-1}$]", "c$_{total,annuity}$\n[kg$\cdot$a$^{-1}$]"], legend='check', tight_layout=True)
pcp_plot_b_2045.set_axis_style(color="dimgrey")  # , alpha=0.5)
pcp_plot_b_2045.add(sensitivity_year_2045_pos_sys_b_pcp,
                    alpha=0.3)  # alpha=0.3
pcp_plot_b_2045.add(sensitivity_year_2045_pos_sys_b_pcp[0], alpha=0.9,
                    color="darkred", linewidth=2.5, label="min. c$_{total,annuity}$")
pcp_plot_b_2045.add(sensitivity_year_2045_pos_sys_b_pcp[1], alpha=0.9,
                    color='olivedrab', linewidth=2.5, label="min. CO$_{2,äq}$")
pcp_plot_b_2045.add(sensitivity_year_2045_pos_sys_b_pcp[sensitivity_year_2045_pf_sys_b_tradeoff],
                    linestyle='-.', alpha=0.5, color='darkcyan', linewidth=3, label="eq. w. Trade-Off")
pcp_plot_b_2045.show()

plt.savefig(result_path + '\\PCP_Sys_B_2045_{}.png'.format(dt.date.today()),
            dpi=300, format='png')

"""
# 2030 & 2045 - w/o CHP
pcp_plot = PCP(cmap='cividis', figsize=(15,8), labels=["P$_{PVT,el}$", "E$_{BESS}$", "V$_{Storage}$", "Q$_{ASHP}$", "t$_{biv}$", "CO$_{2,äq}$", "c$_{total,annuity}$"], legend='check', tight_layout=True) 
pcp_plot.set_axis_style(color="dimgrey") #, alpha=0.5)
# 2030
pcp_plot.add(sensitivity_year_2030_pos_sys_a_pcp[0], alpha=0.9, color="darkred", linewidth=2.5, label="min. c$_{total,annuity}$ - 2030")
pcp_plot.add(sensitivity_year_2030_pos_sys_a_pcp[1], alpha=0.9, color='olivedrab', linewidth=2.5, label="min. CO$_{2,äq}$ - 2030")
pcp_plot.add(sensitivity_year_2030_pos_sys_a_pcp[sensitivity_year_2030_pf_sys_a_tradeoff], alpha=0.5, color='darkcyan',linewidth=2.5, label="Trade-Off Solution -2030")
# 2045
pcp_plot.add(sensitivity_year_2045_pos_sys_a_pcp[0], alpha=0.9, color="darkred", linewidth=2.5, linestyle='-.', label="min. c$_{total,annuity}$ - 2030")
pcp_plot.add(sensitivity_year_2045_pos_sys_a_pcp[1], alpha=0.9, color='olivedrab', linewidth=2.5, linestyle='-.', label="min. CO$_{2,äq}$ - 2030")
pcp_plot.add(sensitivity_year_2045_pos_sys_a_pcp[sensitivity_year_2045_pf_sys_a_tradeoff], linestyle='-.', alpha=0.5, color='darkcyan',linewidth=2.5, label="Trade-Off Solution -2030")
pcp_plot.show()
"""

"___Plotting: 3D plots for electricitcy Price sensitivities"

# Plot Sys Configuration A

# Convert to ndarray
x = sensitivity_elec_price_sys_a["Overall Costs [€/a]"].to_numpy()
y = sensitivity_elec_price_sys_a["Electricity Price [€/kWh]"].to_numpy()
z = sensitivity_elec_price_sys_a["CO2äq [kg/a]"].to_numpy()

surf_sys_a = plt.figure(figsize=(10, 10)).add_subplot(
    projection='3d')  # initiate figure
# Use trisurf by matplotlib
surf_sys_a.plot_trisurf(x, y, z, cmap='RdYlGn_r', linewidth=0)
# Colorbar
norm = mpl.colors.Normalize(vmin=round(sensitivity_elec_price_sys_a["CO2äq [kg/a]"].min()-2), vmax=round(
    sensitivity_elec_price_sys_a["CO2äq [kg/a]"].max(), -2))  # Norm colors for colorbar in dependecy of CO2 min, max

ticks = np.linspace(start=round(sensitivity_elec_price_sys_a["CO2äq [kg/a]"].min(), -2),
                    stop=round(
                        sensitivity_elec_price_sys_a["CO2äq [kg/a]"].max(), -2),
                    num=6
                    )  # Colorbar ticks with linspace

ticks = [500, 1000, 1500, 2000, 2500, 3000, 3500]  # looks better ...

# bounds = [round(sensitivity_elec_price_sys_a["CO2äq [kg/a]"].min(),-2), round(sensitivity_elec_price_sys_a["CO2äq [kg/a]"].max(),-2)] #Colorbar bounds

surf_sys_a.view_init(15, -70)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn_r'), shrink=0.5,
             label="CO$_{2,äq}$ [kg$\cdot a^{-1}$]", ticks=ticks)  # Set Colorbar

# Axis
plt.ylim(price_range[0], price_range[-1])
plt.xticks(rotation=30)
plt.xlabel(
    "Total Annuity System Costs  [€$\cdot a^{-1}$]", fontsize=12, labelpad=25)
plt.ylabel("Electricity Price [€$\cdot kWh^{-1}$]", fontsize=12)
plt.show()

plt.savefig(result_path + '\\PF_Sys_A_Sensitivity_Electricity_Price_{}.png'.format(
    dt.date.today()), dpi=300, format='png')

# Plot Sys Configuration B

# Convert to ndarray
x = sensitivity_elec_price_sys_b["Overall Costs [€/a]"].to_numpy()
y = sensitivity_elec_price_sys_b["Electricity Price [€/kWh]"].to_numpy()
z = sensitivity_elec_price_sys_b["CO2äq [kg/a]"].to_numpy()

surf_sys_a = plt.figure(figsize=(10, 10)).add_subplot(
    projection='3d')  # initiate figure
# Use trisurf by matplotlib
surf_sys_a.plot_trisurf(x, y, z, cmap='RdYlGn_r', linewidth=0)

# Colorbar
norm = mpl.colors.Normalize(vmin=round(sensitivity_elec_price_sys_b["CO2äq [kg/a]"].min()-2), vmax=round(
    sensitivity_elec_price_sys_b["CO2äq [kg/a]"].max(), -2))  # Norm colors for colorbar in dependecy of CO2 min, max
ticks = [500, 1000, 1500, 2000, 2500]  # looks better ...

surf_sys_a.view_init(15, -70)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn_r'), shrink=0.5,
             label="CO$_{2,äq}$ [kg$\cdot a^{-1}$]", ticks=ticks)  # Set Colorbar

# Axis
plt.ylim(price_range[0], price_range[-1])
plt.xticks(rotation=30)
plt.xlabel(
    "Total Annuity System Costs  [€$\cdot a^{-1}$]", fontsize=12, labelpad=25)
plt.ylabel("Electricity Price [€$\cdot kWh^{-1}$]", fontsize=12)
plt.show()

plt.savefig(result_path + '\\PF_Sys_B_Sensitivity_Electricity_Price_{}.png'.format(
    dt.date.today()), dpi=300, format='png')

"___Plotting: 3D plots for hydrogen Price sensitivities"

# Plot Sys Configuration A

# Convert to ndarray
x = sensitivity_hydrogen_price_sys_a["Overall Costs [€/a]"].to_numpy()
y = sensitivity_hydrogen_price_sys_a["Hydrogen Price [€/kg]"].to_numpy()
z = sensitivity_hydrogen_price_sys_a["CO2äq [kg/a]"].to_numpy()

surf_sys_a_h2 = plt.figure(figsize=(10, 10)).add_subplot(
    projection='3d')  # initiate figure
surf_sys_a_h2.plot_trisurf(x, y, z, cmap='RdYlGn_r',
                           linewidth=0)  # Use trisurf by matplotlib
# Colorbar
norm = mpl.colors.Normalize(vmin=round(sensitivity_hydrogen_price_sys_a["CO2äq [kg/a]"].min()-2), vmax=round(
    sensitivity_hydrogen_price_sys_a["CO2äq [kg/a]"].max(), -2))  # Norm colors for colorbar in dependecy of CO2 min, max

ticks = np.linspace(start=round(sensitivity_hydrogen_price_sys_a["CO2äq [kg/a]"].min(), -2),
                    stop=round(
                        sensitivity_hydrogen_price_sys_a["CO2äq [kg/a]"].max(), -2),
                    num=6
                    )  # Colorbar ticks with linspace

ticks = [500, 1000, 1500, 2000, 2500, 3000, 3500]  # looks better ...

# bounds = [round(sensitivity_elec_price_sys_a["CO2äq [kg/a]"].min(),-2), round(sensitivity_elec_price_sys_a["CO2äq [kg/a]"].max(),-2)] #Colorbar bounds

surf_sys_a_h2.view_init(15, -70)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn_r'), shrink=0.5,
             label="CO$_{2,äq}$ [kg$\cdot a^{-1}$]")  # , ticks=ticks ) # Set Colorbar

# Axis
plt.ylim(price_range_h2[0], price_range_h2[-1])
plt.xticks(rotation=30)
plt.xlabel(
    "Total Annuity System Costs  [€$\cdot a^{-1}$]", fontsize=12, labelpad=25)
plt.ylabel("H2 Price [€$\cdot kg_{H2,Hs}$$^{-1}$]", fontsize=12)
plt.show()

plt.savefig(result_path + '\\PF_Sys_A_Sensitivity_Hydrogen_Price_{}.png'.format(
    dt.date.today()), dpi=300, format='png')


# Plot Sys Configuration B

# Convert to ndarray
x = sensitivity_hydrogen_price_sys_b["Overall Costs [€/a]"].to_numpy()
y = sensitivity_hydrogen_price_sys_b["Hydrogen Price [€/kg]"].to_numpy()
z = sensitivity_hydrogen_price_sys_b["CO2äq [kg/a]"].to_numpy()

surf_sys_b_h2 = plt.figure(figsize=(10, 10)).add_subplot(
    projection='3d')  # initiate figure
surf_sys_b_h2.plot_trisurf(x, y, z, cmap='RdYlGn_r',
                           linewidth=0)  # Use trisurf by matplotlib
# Colorbar
norm = mpl.colors.Normalize(vmin=round(sensitivity_hydrogen_price_sys_b["CO2äq [kg/a]"].min()-2), vmax=round(
    sensitivity_hydrogen_price_sys_b["CO2äq [kg/a]"].max(), -2))  # Norm colors for colorbar in dependecy of CO2 min, max

ticks = np.linspace(start=round(sensitivity_hydrogen_price_sys_b["CO2äq [kg/a]"].min(), -2),
                    stop=round(
                        sensitivity_hydrogen_price_sys_b["CO2äq [kg/a]"].max(), -2),
                    num=6
                    )  # Colorbar ticks with linspace

ticks = [500, 1000, 1500, 2000, 2500, 3000, 3500]  # looks better ...

# bounds = [round(sensitivity_elec_price_sys_a["CO2äq [kg/a]"].min(),-2), round(sensitivity_elec_price_sys_a["CO2äq [kg/a]"].max(),-2)] #Colorbar bounds

surf_sys_b_h2.view_init(15, -70)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn_r'), shrink=0.5,
             label="CO$_{2,äq}$ [kg$\cdot a^{-1}$]")  # , ticks=ticks ) # Set Colorbar

# Axis
plt.ylim(price_range_h2[0], price_range_h2[-1])
plt.xticks(rotation=30)
plt.xlabel(
    "Total Annuity System Costs  [€$\cdot a^{-1}$]", fontsize=12, labelpad=25)
plt.ylabel("H2 Price [€$\cdot kg_{H2,Hs}$$^{-1}$]", fontsize=12)
plt.show()

plt.savefig(result_path + '\\PF_Sys_B_Sensitivity_Hydrogen_Price_{}.png'.format(
    dt.date.today()), dpi=300, format='png')

"___Plotting: Annex & Other___"

"""
def load_duration_curve(self,
                        x_axis,
                        load
                        ):
    
   # Load Duration Curve Heat Demand Over Temperature
    load_duration_curve = pd.DataFrame(data={'{}'.format(temp_data["Temperature [°C]"].name): temp_data["Temperature [°C]"], 
                                             '{}'.format(building_demand["Sum Space Heating Net [kWh]"].name): building_demand["Sum Space Heating Net [kWh]"]}) 
    
    load_duration_curve['interval'] = 1
    
    load_duration_curve_sorted = load_duration_curve.sort_values(by=['{}'.format(building_demand["Sum Space Heating Net [kWh]"].name)], ascending = False)
    load_duration_curve_sorted['Annual Hours [h]'] = load_duration_curve_sorted['interval'].cumsum()
    
    temp_sorted = load_duration_curve_sorted.sort_values(by=['{}'.format(load_duration_curve_sorted["Temperature [°C]"].name)], ascending = True)
    
    # Load Duration Curve - Load over Temperature
    plt.figure()
    plt.plot(temp_sorted['Temperature [°C]'], load_duration_curve_sorted["Sum Space Heating Net [kWh]"])
    plt.ylabel('{}'.format(building_demand["Sum Space Heating Net [kWh]"].name))
    plt.xlabel('{}'.format(load_duration_curve_sorted["Temperature [°C]"].name))
    plt.xlim(load_duration_curve_sorted["Temperature [°C]"].min(), 20)
    
    # Load Duration Curve - Load over Duration
    plt.figure()
    plt.plot(load_duration_curve_sorted['duration'], load_duration_curve_sorted["Sum Space Heating Net [kWh]"])
    plt.ylabel('{}'.format(building_demand["Sum Space Heating Net [kWh]"].name))
    plt.xlabel('{}'.format(load_duration_curve_sorted['duration'].name))
"""

"___H2_CHP_Efficiency_Curve___"
"""
h2_chp_power = [0, 0.020, 0.032, 0.050, 0.067, 0.085, 1]
h2_chp_eff = [0, 55, 58, 59.5, 60, 59.7, 45]

power_spline = np.linspace(0, 1, 500)
spline_func = make_interp_spline(h2_chp_power, h2_chp_eff, k=3)
eff_spline = spline_func(power_spline)


plt.figure()
plt.rcParams.update({'font.size': 20})
plt.plot(h2_chp_power, h2_chp_eff, 'o-', ms= 0, lw=3, c='darkred')
plt.ylabel(r"$\eta$$_{el}$ [%]")
plt.xlabel("Electric Power normed [-]")
plt.title(r'Electric system efficiency curve PEMFC-CHP')
plt.xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlim(0, 1)
plt.ylim(0, 75)


"___HV_Sensitivity_on_Generations___"
"""
generations = [20, 40, 60, 80, 100]
hv_rel = [0.9890, 0.9971, 0.9999, 0.9999, 1]
pf_text = "Algorithm: NSGA-II\nPopulation: 50"

plt.figure(figsize=(9, 5))
plt.rcParams.update({'font.size': 15})
plt.plot(generations, hv_rel, 'o-', ms=7, lw=3, c='darkred')
plt.ylabel(r"HV (rel. to HV$_{gen. 100}$ [-]")
plt.xlabel("Number of Generations [-]")
# plt.title(r'Sensitivity of HV over (NSGA-II, Pop. 50)')
# plt.xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlim(0, 120)
plt.ylim(0.988, 1.001)
plt.grid(which='major', axis='y', alpha=0.4)
plt.text(0.73, 0.09, pf_text, transform=ax.transAxes,  fontdict={
         'fontsize': 14, 'horizontalalignment': 'left', 'verticalalignment': 'top'}, backgroundcolor='white')

print("The displayed Illustrations are exported to:\n\n", result_path)
