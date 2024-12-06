# -*- coding: utf-8 -*-
"""
- Created within the Framework of a masterthesis @
faculty of Process Engineering, Energy and Mechanical Systems
of the Cologne University of Applied Sciences -

-> Secondary Programm of Green_Building_Opt.py
for the parametrization of energy system components & cost calculation
@author: marius bartkowski

Contact:
marius.bartkowski@magenta.de or
marius.bartkowski@smail.th-koeln.de
"""

# Libs
import math
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

# Global Functions
def q_loss_regression(x, c1):   # Root Function for regression of q_loss of warm water storage in standby 
    return c1*np.sqrt(x)

def heatflux_temperature_regression(x, c1, c2): # Qudratic Function for regression of the BIPVT pant air heatflux temperature difference to heated module temperature
    return c1*x**2 + c2*x

def annuity(i0, n, dr):

    """
    Is used to determine dynamic fuel prices. (linear increase/decrease of fuel price by a certain price change rate)

    Parameters:
            -----------

                i0 : numeric
                    CAPEX or specific costs @ t=0

                n: numeric
                    runtime or invetsigated time period

                dr: numeric
                    discount rate

    Output:
            -----------

                c_a: numeric
                    annual annuity costs

                c: numeric
                    total annuity costs

                c_t: DataFrame
                    annuity costs at timestep t
    """

    c_a = (i0 * (1 + dr)**n)/n
    c = i0 * (1 + dr)**n

    annuity_absosulte = [c, c_a]

    annuity_timeseries = []
    t=0

    while t < n:
        c_t = [t, (i0 * (1 + dr)**t)]
        annuity_timeseries.append(c_t)
        t+=1

    annuity_timeseries = pd.DataFrame(data=annuity_timeseries, columns=["Year", "Costs [€]"])

    return annuity_absosulte, annuity_timeseries


def annuity_factor(n, dr):

    """
    Estimates annuity factor with discount rate dr over lifetime n

    Parameters:
            -----------

                k : numeric
                    CAPEX or specific costs @ t=0

                n: numeric
                    runtime or invetsigated time period

                dr: numeric
                    discount rate

    Output:
            -----------

                a_f: numeric
                    annuity factor [-]

    """

    a_f = ((dr*(1+dr)**n)/(((1+dr)**n)-1))

    return a_f


def overall_costs(components, energy_system_results_electrical, energy_system_results_thermal, specs_eco, fuel, specs_renovation, building):

    """
    Estimates annuity- and overall system costs over lifetime, based on timeseries simulation results
    """

    # Estimating hydrogen, natural gas and electricity price + heat pump tarif considering the rate of price change
    h2_price, h2_price_t = annuity(i0=specs_eco["h2 price"], n=specs_eco["Period"], dr=specs_eco["h2 price dr"])
    ng_price, ng_price_t = annuity(i0=specs_eco["ng price"], n=specs_eco["Period"], dr=specs_eco["ng price dr"])
    elec_price, elec_price_t = annuity(i0=specs_eco["electricity price"], n=specs_eco["Period"], dr=specs_eco["electricity price dr"])
    hp_tarif, hp_tarif_t = annuity(i0=specs_eco["HP Tarif"], n=specs_eco["Period"], dr=specs_eco["HP Tarif Price dr"])

    # Calculating amounts of electric energy In- and Out of the system (per Year)
    chp_electric_power = energy_system_results_electrical["P_el CHP [kW el.]"].sum() # Sum of electric generation of CHP
    pvt_feed_in_elec = energy_system_results_electrical["Grid [kW]"][energy_system_results_electrical["Grid [kW]"] < 0].sum() # electricity fed into the grid can be estimated by summing up "negative" Grid Power
    grid_cover_heat_pump_share = energy_system_results_electrical["Heat Pump Grid Cover Share [kW]"].sum()
    grid_cover = energy_system_results_electrical["Grid [kW]"][energy_system_results_electrical["Grid [kW]"] > 0].sum() - grid_cover_heat_pump_share   # Vice versa, the "positive" Grid power equals the net cover - the share of the heat pump grid cover

    # Calculating revenues or costs/revenues in dependency of the feed-in, in-house consumption or grid electric energy over the period of investigation. Costs are negative, Revenues are positive
    revenue_feed_in_elec = (elec_price_t["Costs [€]"] * pvt_feed_in_elec*(-1)).sum()
    costs_grid_cover = (grid_cover * elec_price_t["Costs [€]"]).sum() * (-1)
    costs_grid_cover_hp = ((grid_cover_heat_pump_share * hp_tarif_t["Costs [€]"]).sum()) * (-1)

    # Calculating CHP surcharge in dependency of full load hours
    sum_chp_p_el = energy_system_results_electrical["P_el CHP [kW el.]"].sum()
    capacity_chp = components.iloc[5]["Capacity"]
    if capacity_chp == 0: # in case capacity is zero the divison later would be by zero
        sum_chp_p_el = 0
        capacity_chp = 1
    chp_full_load_hours = sum_chp_p_el / capacity_chp # CHP electrical generation / CHP installed capacity electric
    
    # Setting surcharged Full load hours for CHP System
    if 2023 <= specs_eco["Year of Investigation"] < 2030: # 4000 Full Load hours for 2023 - 2030
        full_load_hours_covered = 4000
    elif specs_eco["Year of Investigation"] >= 2030:    # 2500 Full Load hours since 2030
        full_load_hours_covered = 2500
    else:   # 5000 full load hours for every other year (< 2023)
        full_load_hours_covered = 5000

    max_full_load_hours = 30000 # Quantity of full load hours covered by KWKG:2023
    if chp_full_load_hours > full_load_hours_covered: # If Full Load hours of CHP system exceed surcharged full load hours, cap surcharged full load hours
        years_of_surcharge = max_full_load_hours / full_load_hours_covered # Years of surcharge in dependency of maximum fullload hours

        if years_of_surcharge > specs_eco["Period"]: # If Years of surcharge exceed Investigated Period, Estimate Surcharge for investigated period
            years_of_surcharge = specs_eco["Period"]
        else:
            pass

        surcharge_chp_in_house_consumption = full_load_hours_covered * components.iloc[5]["Capacity"] * specs_eco["CHP surcharge"] * years_of_surcharge
    else: # If Full Load hours of CHP system lie below surcharged full load hours -> Calculate with system full load hours without capping
        years_of_surcharge = max_full_load_hours / full_load_hours_covered # Years of surcharge in dependency of maximum fullload hours
        if years_of_surcharge > specs_eco["Period"]: # If Years of surcharge exceed Investigated Period, Estimate Surcharge for investigated period
            years_of_surcharge = specs_eco["Period"]
        else:
            pass

        surcharge_chp_in_house_consumption = chp_full_load_hours * components.iloc[5]["Capacity"] * specs_eco["CHP surcharge"] * years_of_surcharge

    # Total energy Cost (bilancy)
    electric_energy_costs_total = costs_grid_cover + costs_grid_cover_hp + revenue_feed_in_elec + surcharge_chp_in_house_consumption # + revenue_pv_in_house_consumption

    # DataFrame for results
    electric_energy_costs = [("PV Feed in", pvt_feed_in_elec*specs_eco["Period"], revenue_feed_in_elec),
                             ("Grid Cover w/o HP", grid_cover*specs_eco["Period"], costs_grid_cover),
                             ("Grid Cover HP", grid_cover_heat_pump_share*specs_eco["Period"], costs_grid_cover_hp),
                             #("In-House Consumption", pv_in_house_consumption*specs_eco["Period"], revenue_pv_in_house_consumption),
                             ("CHP el. Surcharge", chp_electric_power*specs_eco["Period"], surcharge_chp_in_house_consumption),
                             ("Bilancy", 0, electric_energy_costs_total)
                             ]

    electric_energy_costs = pd.DataFrame(data=electric_energy_costs, columns=["type", "E [kWh]", "Costs/Revenue over runtime [€]"])
    electric_energy_costs.set_index(electric_energy_costs.type, inplace=True) # Set index as of type

    # Setting components name as Index for the components DataFrame
    components.set_index(components.component, inplace=True)

    costs = []


    # Only for Sensitivity year of investigation 2030, 2045
    cost_reductions = [("PVT", 0),             # as respected already in end consumer price
                      ("Fan", 0),
                      ("Battery", 43.5/100),
                      ("HP", 22/100),
                      ("H2 Boiler", 0),
                      ("NG Boiler", 0),
                      ("H2 CHP", 46.1/100),
                      ("EMC", 0),
                      ("Water Storage", 0),
                      ("Renovation", 0)
                      ]

    cost_reductions = pd.DataFrame(data=cost_reductions, columns=["component", "reduction"])

    for component in range(len(components)):

        # CAPEX as annuities [€/a]
        a_f = annuity_factor(n=components["Lifetime"][component], dr=specs_eco["credit dr"]) # Calculating Annuity factor in dependency of component lifetime

        # Investment sum in dependency of component capacity by Cost function or simple specific Investment cost
        try: # try to estimate i0 by cost function via accessing tuple. As simple float values can't be called, investment sums without cost functions are estimated by statement for 'except'.

            if (components["component"][component] == "Battery") or (components["component"][component] == "H2 CHP"): # Investment sums for cost functions: kfix + kvar * capacity (H2 CHP, Li-Ion Battery)
                i0 =  specs_eco["CAPEX {}".format(components["component"][component])][0] + (components["Capacity"][component]*specs_eco["CAPEX {}".format(components["component"][component])][1])

                if components["Capacity"][component] == 0: #For example for CHP in case of system config. A
                    i0 = 0
                else:
                    pass

                if (components["component"][component] == "H2 CHP") and (specs_eco["Period"] > specs_eco["Lifetime H2 CHP"]): # Respect stack replacement for H2 Fuel cell CHP after lifetime
                    i0 = i0 + (i0 * 0.2) # In magnitude of 20 % of Investment Sum
                else:
                    pass

            elif (components["component"][component] == "EMS"):
                if components["Capacity"][component] > 1: #For example for CHP in case of system config. A
                    i0 =  specs_eco["CAPEX {}".format(components["component"][component])]
                   
                    
                elif components["Capacity"][component] <= 1:
                    i0 = 0
                    
                    
            elif (components["component"][component] == "Renovation"):
                renov_invest_roof =     ((specs_renovation["roof renov_invest"] * building["area_roof"] * 2) + specs_renovation["planning_cost"] + specs_renovation["scaffolding_cost"]) * specs_renovation["baupreis_index"] # plus Pauschale/Fixkosten für Gerüst und Planung
                renov_invest_windows =  renov_invest_roof + ((specs_renovation["windows renov_invest"] * building["amount_windows"]) * specs_renovation["baupreis_index"]) # jeder Kostenbestandteil wird mit Baupreisindex auf Jahr 2023 umgerechnet
                renov_invest_walls =    renov_invest_windows + (((specs_renovation["walls renov_invest"] * building["area_walls_exterior"])) * specs_renovation["baupreis_index"])
                renov_invest_cellar =   renov_invest_walls + ((specs_renovation["cellar renov_invest"] * building["area_house_base"]) * specs_renovation["baupreis_index"])
                renov_invest_full =     renov_invest_cellar + (((specs_renovation["ventilation renov_invest"] * (building["area_effective_living"] / 1.2))) * specs_renovation["baupreis_index"])
                if components["Capacity"][component] < 1:
                    i0 = 0
                elif components["Capacity"][component] < 2:
                    i0 = renov_invest_roof
                elif components["Capacity"][component] < 3:
                    i0 = renov_invest_windows
                elif components["Capacity"][component] < 4:
                    i0 = renov_invest_walls
                elif components["Capacity"][component] < 5:
                    i0 = renov_invest_cellar
                elif components["Capacity"][component] <= 6:
                    i0 = renov_invest_full

            else: # Investment sums for cost functions: a*capacity^b (Boiler, HP Water Storage)
            # invalid values encountered in double scalars
                i0 =  components["Capacity"][component] * (specs_eco["CAPEX {}".format(components["component"][component])][0] * components["Capacity"][component]**specs_eco["CAPEX {}".format(components["component"][component])][1])

            # Convert i0 from cost functions to present year to €2023 by respecting inflation. And cost decrease.
            i0 = i0 * ((1+specs_eco["inflation dr"])**(specs_eco["Year of Investigation"] - specs_eco["CAPEX {}".format(components["component"][component])][2])) * (1+(specs_eco["CAPEX {}".format(components["component"][component])][3]/100))
            
        except: # Investment Sum using float values capacity * spec. Ivestment costs

            if components["component"][component] == "Renovation":
                pass
            elif components["component"][component] == "EMS":
                pass
            else:
                i0 = (components["Capacity"][component] * specs_eco["CAPEX {}".format(components["component"][component])])
        
        
        if specs_eco["Year of Investigation"] >= 2030:
            i0 = i0 * (1-cost_reductions["reduction"][component])
        else:
            pass

        if components["component"][component] == "H2 Boiler": # Add 300 € für H2 Retroffitting of conventional technology in case of H2 Boiler
            capex_a = (i0 + 300) * a_f
        else:
            capex_a = i0 * a_f

        # OPEX without expension
        try: # try, because not every component has OPEX
            c_fix = i0 * specs_eco["OPEX fix {}".format(components["component"][component])]

        except:
            c_fix = 0

        costs.append([i0, a_f, capex_a, c_fix, 0, capex_a*specs_eco["Period"], c_fix*specs_eco["Period"], 0]) # Append results list
        #print("Components:\n " ,component, "\nCosts:\n", costs)
        
    # creating Data Frame with results from CAPEX/OPEX calculation
    costs = pd.DataFrame(data=costs, columns=["Investment Sum [€]", "Annuity Factor","CAPEX annuities [€/a]","OPEX fix [€/a]","OPEX var [€/a]","CAPEX [€]","OPEX fix [€]","OPEX var [€]"])

    costs.set_index(components.component, inplace=True) # Set the same index as of components to enable joining on index
    components.drop(columns="component", inplace=True)  # Drop unnecessary columns
    components_joined = components.join(costs, how='outer') #Join DataFrame "components" & "costs"

    # Calculate variable OPEX of H2/NG Boiler & H2 CHP
    if fuel == "NG":
        boiler_opex_var = (ng_price_t["Costs [€]"] * energy_system_results_thermal["Cons. Boiler [kg]"].sum()).sum()
    else:
        boiler_opex_var = (h2_price_t["Costs [€]"] * energy_system_results_thermal["Cons. Boiler [kg]"].sum()).sum()

    chp_opex_var = (h2_price_t["Costs [€]"] * energy_system_results_thermal["H2 cons. CHP [kg H2]"].sum()).sum()

    components_joined.at['Boiler', 'OPEX var [€]'] = boiler_opex_var
    components_joined.at['Boiler', 'OPEX var [€/a]'] = boiler_opex_var/specs_eco["Period"]

    components_joined.at['H2 CHP', 'OPEX var [€]'] = chp_opex_var
    components_joined.at['H2 CHP', 'OPEX var [€/a]'] = chp_opex_var/specs_eco["Period"]

    price_dynamics = pd.DataFrame(data={'Year':h2_price_t["Year"],
                                        'Hydrogen Price [€]':h2_price_t["Costs [€]"],
                                        'Natural Gas Price [€]':ng_price_t["Costs [€]"],
                                        'Electricity Price [€]':elec_price_t["Costs [€]"],
                                        'Heat Pump Tarif [€]': hp_tarif_t["Costs [€]"]
                                        })

    return components_joined, electric_energy_costs, price_dynamics


# Parametrization - Energy System Components
def heat_flux_temperature_difference(solar_roof_tile_data, irrad_upper):

    """
    Estimates the temperature difference between the cooled module and air-flow of the PV/T sytsem,
    in order to calcualte the source tempeerature for the HP
    """

    heat_flux_temperature_difference = [] # Creating empty list

    while irrad_upper >= 50:

        heat_flux_temperature_difference.append(
            (irrad_upper,
            solar_roof_tile_data["Tmcooling [°C]"][(solar_roof_tile_data["POA Global [W/m2]"] <= irrad_upper) & (solar_roof_tile_data["POA Global [W/m2]"] > irrad_upper-50)].mean(),
            solar_roof_tile_data["Temperature Difference Heat Flux [°C]"][(solar_roof_tile_data["POA Global [W/m2]"] <= irrad_upper) & (solar_roof_tile_data["POA Global [W/m2]"] > irrad_upper-50)].mean()
            )
            )
        irrad_upper-=50

    heat_flux_temperature_difference = pd.DataFrame(heat_flux_temperature_difference)
    heat_flux_temperature_difference.insert(loc=3, column='Heatflux Temperature Difference [-]', value=heat_flux_temperature_difference[2]/heat_flux_temperature_difference[1])

    irrad = heat_flux_temperature_difference[0]

    regression_coefficients = curve_fit(heatflux_temperature_regression, irrad, heat_flux_temperature_difference["Heatflux Temperature Difference [-]"])

    fit=[]

    for i in irrad:
        fit.append(heatflux_temperature_regression(i, regression_coefficients[0][0], regression_coefficients[0][1]))

    r_regression = np.corrcoef(heat_flux_temperature_difference[2], fit)                                 # R value of fit

    """
    plt.rcParams.update({'font.size': 16})
    pf_text= r'y={}'.format(round(regression_coefficients[0][0], 10)) + r'$\cdot$x$^{2}$ + ' + r'{}'.format(round(regression_coefficients[0][1], 5)) + "$\cdot$x" + "\nR$^2$={}".format(round(r_regression[0][1], 6))
    plt.plot(heat_flux_temperature_difference[0], heat_flux_temperature_difference["Heatflux Temperature Difference [-]"])
    plt.plot(heat_flux_temperature_difference[0], fit)
    plt.text(x=1100, y=0.18, s=pf_text,   fontdict={'fontsize': 16, 'horizontalalignment': 'right', 'verticalalignment':'bottom'})
    plt.legend(['Measurement Values for t$_{diff,m}$ / $t_{mod}$','Regression Curve'])
    plt.xlabel("Plane of Array Irradiance [W$\cdot$m$^{-2}$]")
    plt.ylabel("t$_{diff,m}$/$t_{mod}$")
    """

    return regression_coefficients, r_regression


def q_loss_specific_din12831(combined_heat_and_power):

    """
    Regression of the specific heat loss for the warm water storage by the data given in Annex B of DIN 12831
    """

    q_loss_standby = {'WW Storage Volume Brutto': [5,30,50,80,100,120,150,200,300,400,500,600,800,1000,1250,1500,2000],
                      'q loss Standby': [0.35,0.60,0.78,0.98,1.10,1.20,1.35,1.56,1.91,2.20,2.46,2.69,3.11,3.48,3.89,4.26,4.92]}

    if combined_heat_and_power == True:
        n_flanges = 7   # Two additional flanges for CHP Connestion to storage
    else:
        n_flanges = 5   # Two additional flanges for Heat Pump Connection, 1 for reheating by gas boiler and another 2 for DHW heat exchanger

    q_loss_standby = pd.DataFrame(data=q_loss_standby)
    q_loss_standby['q loss Standby'] = (q_loss_standby['q loss Standby'] + (0.1 * n_flanges))/24      # /24 for hourly values [kWh_th/h]
    # a set of inlet and outflow pipes/flanges are increasing the heat loss by 0.1 kWh/d

    regression_coefficients = curve_fit(q_loss_regression, q_loss_standby["WW Storage Volume Brutto"],  q_loss_standby["q loss Standby"])

    fit=[]

    for i in q_loss_standby["WW Storage Volume Brutto"]:
        fit.append(q_loss_regression(i, regression_coefficients[0][0]))

    r_regression = np.corrcoef(q_loss_standby["q loss Standby"], fit)                                 # R value of fit

    """
    plt.rcParams.update({'font.size': 16})
    pf_text= r'y={}$\cdot$'.format(round(regression_coefficients[0][0], 4)) + "$\sqrt{V_{St}}$" + "\nR$^2$={}".format(round(r_regression[0][1], 6))
    plt.figure()
    plt.plot(q_loss_standby["WW Storage Volume Brutto"], q_loss_standby["q loss Standby"])
    plt.plot(q_loss_standby["WW Storage Volume Brutto"], fit)
    plt.text(x=2000, y=4, s=pf_text,   fontdict={'fontsize': 16, 'horizontalalignment': 'right', 'verticalalignment':'top'})
    plt.legend(['Reference Values','Regression'])
    plt.xlabel("Warm Water Storage Volume [L]")
    plt.ylabel("q$_{St,Loss}$ [kWh/d]")
    plt.title("Warm Water Storage specific Heat Loss Regression")
    """

    return regression_coefficients , r_regression

def autarky_rate(electricity_consumption_sum, thermal_consumption_sum, grid_electricity_consumption_sum, gas_consumption_sum):

    """
    Estimates autarky rate of the system
    """

    grid_electricity_consumption_sum[grid_electricity_consumption_sum < 0] = 0

    # Autarky rate as in Quaschning, Regenerative Energiesysteme (2021) S.282
    autarky = (electricity_consumption_sum + thermal_consumption_sum - grid_electricity_consumption_sum - gas_consumption_sum)/(electricity_consumption_sum + gas_consumption_sum)

    return autarky


def co2_emission_timeline(grid_cover, co2_data, start_year, n):

    """
    Estimates CO2 emission timeline and system CO2 emissions
    """

    # CO2 Emission Timeline
    x_co2_year = [2021, 2030, 2045]             # Years of data for interpolation
    y_co2_emissionfactor= [428, 152.4, 3.9]      # CO2 Emissionfactor for 2021 CO2äq according to UBA 05/2022, for 2030 acccording to KN2045 Agora Energiewende. (98 Mio. t Co2äq/643 TWh Gross electricity Produktion)

    energy_system_lifetime = np.arange(start=start_year, stop=start_year+n, step=1)                             # Arrange numpy array for every year of the energy system lifetime/period of investigation
    co2_energy_mix_years = np.interp(energy_system_lifetime, x_co2_year, y_co2_emissionfactor)   # Interpolate co2 Emissionfactors based on x and y for every year of lifetime

    co2_data[:,[1]].max()   # Max Value of CO2 Emissionfactor
    co2_data_normed = co2_data[:,[1]]/co2_data[:,[1]].max()     # Normed CO2 Timeseries

    co2_energy_timelines = []   # Creating empty array

    for years in range(len(co2_energy_mix_years)):  # iterate through all years of investigation

        co2_emissionfactor_mean=0   # Init variable
        co2_max_init = co2_energy_mix_years[years] # Start with mean CO2 value corresponding to the specific year

        while co2_emissionfactor_mean < co2_energy_mix_years[years]:        # As long as the mean CO2 emission factor from the generated time series is below the actual mean CO2 Emissionfactor from that year, the maximum co2 value is increased. Loop is executed again
            co2_emissionfactor_timeline = co2_data_normed*co2_max_init      # Generating CO2 Emission Series with maximum CO2 value
            co2_emissionfactor_mean = co2_emissionfactor_timeline.mean()    # Estimating MEan CO2 EMissionfactor from Timerseries
            co2_max_init+=1                                                 # Increase CO2 max init

        co2_energy_timelines.extend(co2_emissionfactor_timeline.flatten())            # Extending Timeline Array

    co2_energy_timelines = pd.Series(co2_energy_timelines)     # Create DataFrame
    #plt.plot(co2_energy_timelines)

    # Overall CO2 Emissions over lifetime
    grid_cover_lifetime = []    # Create DataFrame

    i=0
    while i < n:    # Append Grid Cover repeatedly for energy system lifetime
        grid_cover_lifetime.extend(grid_cover)
        i+=1

    grid_cover_lifetime = pd.DataFrame(data={"Grid Cover [kWh]": grid_cover_lifetime}) #Create Data Frame
    grid_cover_lifetime[grid_cover_lifetime < 0] = 0    # Zero all negative Grid Power (Infeed) to consider only indirect CO2e Emissions by Grid Cover

    grid_cover_lifetime = grid_cover_lifetime.squeeze()         # Array Conversation

    system_co2_emission_timeline = (grid_cover_lifetime * co2_energy_timelines)/1000 # Calculate System CO2e Emissions by multiplikation of Grid Cover + CO2e Timeline

    return system_co2_emission_timeline, co2_energy_timelines
