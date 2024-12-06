# The usual suspects
import sys
import numpy as np

from scipy.spatial import distance

# Optimization
import time
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
# from pymoo.visualization.scatter import Scatter

# Algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.age2 import AGEMOEA2

# Optimization Performance metrics
from pymoo.indicators.hv import HV as hyper_volume


def optimize_multi_objective(optimization_mode, optimization_specs, combined_heat_and_power, max_pv_power, temp_data, energy_system_simulation, own_design, n):
    "_____Optimization_____"

    if optimization_mode == True:

        "___Algorithm definition and Parameter_Allocation__"

        if optimization_specs["Algorithm"] == "NSGA2":
            algorithm = NSGA2(pop_size=optimization_specs["Population Size"])
        elif optimization_specs["Algorithm"] == "NSGA3":
            ref_dirs = get_reference_directions(
                "das-dennis", 2, n_partitions=2)
            algorithm = NSGA3(pop_size=optimization_specs["Population Size"],
                              ref_dirs=ref_dirs)
        elif optimization_specs["Algorithm"] == "SMSEMOA":
            algorithm = SMSEMOA(pop_size=optimization_specs["Population Size"])
        elif optimization_specs["Algorithm"] == "AGEMOEA2":
            algorithm = AGEMOEA2(
                pop_size=optimization_specs["Population Size"])
        else:
            print("\nPlease choose one of the following Algorithms:\n\nNSGA2\nUNSGA3\nSMSEMOA\nMOEAD\n\nTerminating Programm")
            sys.exit()

        "___MOP definition & Optimization initiation_____"

        if combined_heat_and_power == True:    # Optimize energy system with CHP operation
            # Defining Problem
            class SystemDesignProblem(Problem):

                def __init__(self):

                    # Initiating lower boundary np.array
                    xl = np.zeros(10)
                    # Initiating upper boundary np.array
                    xu = np.zeros(10)

                    # lower & upper boundary of Solarroof [kW]
                    xl[0] = 1
                    xu[0] = max_pv_power/1000

                    # lower & upper boundary of Solarroof [kW]
                    xl[1] = 1
                    xu[1] = max_pv_power/1000

                    # lower & upper boundary of Battery Storage [kWh]
                    xl[2] = 1
                    xu[2] = 150

                    # lower & upper boundary of Warm Water Storage [L]
                    xl[3] = 200
                    xu[3] = 10000

                    # lower & upper boundary of Heat Pump Nominal Power [kWth]
                    xl[4] = 1
                    xu[4] = 150

                    # lower & upper boundary of Bivalency Point [°C]
                    xl[5] = temp_data["Temperature [°C]"].min()
                    xu[5] = 15

                    # lower & upper boundary of CHP electrical capacity [kW]
                    xl[6] = 0.5
                    xu[6] = 10

                    # 0 to 1 is NG and 1 to 2 is H2
                    xl[7] = 0
                    xu[7] = 2

                    # lower & upper boudary of Electrical Vehicle (where <1 represents simulation without EMS, and >= 1 with EMS)
                    xl[8] = 0
                    xu[8] = 2

                    # lower & upper boundary of renovation cases
                    xl[9] = 0
                    xu[9] = 6

                    # Define number of variables, number of objects, constraints, lower and upper boundaries, respectivly
                    super().__init__(n_var=10, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

                # Evaluate System designs
                def _evaluate(self, designs, out, *args, **kwargs):
                    print("\n")
                    # Output of objective values
                    out["F"] = energy_system_simulation(designs)
                    print("\n")

            # set problem as defined Problem
            problem = SystemDesignProblem()

            # Start time at calculation start
            start_time = time.time()
            optimization_results = minimize(problem,                                        # Initiate solver, here as minimization problem
                                            algorithm,
                                            ('n_gen',
                                             optimization_specs["Generations"]),
                                            seed=optimization_specs["Seed"],
                                            save_history=False,
                                            verbose=True)

            # Calculation Time in minutes
            calculation_time = round(((time.time() - start_time)/60), 2)

            # calculate a hash to show that all executions end with the same result
            # print("\nOverall System costs\tOverall Co2 Emissions")
            # print(optimization_results.F.T)                                                 # print optimization results

        elif combined_heat_and_power == False:    # Optimize energy system without CHP operation

            # Defining Problem
            class SystemDesignProblem(Problem):

                def __init__(self):

                    # Initiating lower boundary np.array
                    xl = np.zeros(9)
                    # Initiating upper boundary np.array
                    xu = np.zeros(9)

                    # lower & upper boundary of Solarroof [kW]
                    xl[0] = 1
                    xu[0] = max_pv_power/1000

                    # lower & upper boundary of Solarroof [kW]
                    xl[1] = 1
                    xu[1] = max_pv_power/1000

                    # lower & upper boundary of Battery Storage [kWh]
                    xl[2] = 1
                    xu[2] = 150

                    # lower & upper boundary of Warm Water Storage [L]
                    xl[3] = 200
                    xu[3] = 10000

                    # lower & upper boundary of Heat Pump Nominal Power [kWth]
                    xl[4] = 1
                    xu[4] = 150

                    # lower & upper boundary of Bivalency Point [°C]
                    xl[5] = temp_data["Temperature [°C]"].min()
                    xu[5] = 15

                    # 0 to 1 is NG and 1 to 2 is H2
                    xl[6] = 0
                    xu[6] = 2

                    # lower & upper boudary of Electrical Vehicle (where <1 represents simulation without EMS, and >= 1 with EMS)
                    xl[7] = 0
                    xu[7] = 2

                    # lower & upper boundary of renovation cases
                    xl[8] = 0
                    xu[8] = 6

                    # Define number of variables, number of objects, constraints, lower and upper boundaries, respectivly
                    super().__init__(n_var=9, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

                # Evaluate System designs
                def _evaluate(self, designs, out, *args, **kwargs):
                    print("\n")
                    # Output of objective values
                    out["F"] = energy_system_simulation(designs)
                    print("\n")

            # set problem as defined Problem
            problem = SystemDesignProblem()

            # Start time at calculation start
            start_time = time.time()
            optimization_results = minimize(problem,                                        # Initiate solver, here as minimization problem
                                            algorithm,
                                            ('n_gen',
                                             optimization_specs["Generations"]),
                                            seed=optimization_specs["Seed"],
                                            save_history=True,
                                            verbose=True)

            # Calculation Time in minutes
            calculation_time = round(((time.time() - start_time)/60), 2)

            # calculate a hash to show that all executions end with the same result
            # print("\nOverall System costs\tOverall Co2 Emissions")
            # print(optimization_results.F.T)                                                 # print optimization results

        else:
            pass

        "___Performance metrics___"

        # Hypervolume Indicator HV:
        # Reference point for hypervolume indicator. Based on 10,000 kg of CO2 Emissions/a and 50,000 € overall annuity system costs.
        hv_reference_point = np.array([50000, 10000])

        # optimization_results
        # Indicating hv metric with set reference point
        hyper_volume_indicator_metric = hyper_volume(
            ref_point=hv_reference_point)
        hyper_volume_indicator = hyper_volume_indicator_metric(
            optimization_results.F)  # calculating HV for optimization results

        print("\nHypervolume HV:\t", round(
            hyper_volume_indicator, 0))                  # Print res

        # Spacing S:
        # Number of pareto points
        n_pareto_points = optimization_results.F.shape[0]

        # Estimate euclidean distance between all perato points
        euclidean_distance_uncorrected = distance.pdist(
            optimization_results.F, metric='euclidean')
        euclidean_distance_sorted = np.sort(
            euclidean_distance_uncorrected)                         # sort Distances
        # Consider all euclidean distances except the biggest distance. the biggest distance is the distance between the two extremes, which isn't a distance between the nearest member as in [17] and therefore left out
        d = euclidean_distance_sorted[0:-1]

        dm = np.mean(d)   # Mean of e. distance

        # Spacing S as in Goh, C.; Tan, K.: Evolutionary Multi-Objective Optimization in Uncertain Envrionments (2009)
        spacing = (np.sqrt((np.sum(np.square(d-dm)))/n_pareto_points)) / dm

        # Print res
        print("Spacing S:\t\t", round(spacing, 3))

        """ Experimental
        rigd = RMetric(problem=problem,
                       ref_points= np.array([[0,0], [0,0]]),
                       w=None,
                       delta=0.2,
                       pf=optimization_results.F)

        rig = rigd(optimization_results.F)
        """

        energy_system_results_electrical = 0
        energy_system_results_thermal = 0

        return optimization_results, energy_system_results_electrical, energy_system_results_thermal, hyper_volume_indicator, spacing

    # Optimization mode Set False: One single design (Own design, defined at top section) is run and different results are returned by the function
    elif optimization_mode == False:
        result_design, energy_system_results_electrical, energy_system_results_thermal, warm_water_storage_temperatures, components, electric_energy_costs, price_dynamics, system_ops, solar_thermal_energy, ww_storage_balance, heat_pump_energies, h2_boiler_energy, electrical_storage_balance, overall_system_co2_emissions, overall_system_costs, system_co2_emission_timeline, co2_emission_timelines = energy_system_simulation(
            own_design)

        "__Further_Technical_KPIs___"

        jaz_ashp = energy_system_results_thermal["q HP [kW th.]"].sum(
        ) / (energy_system_results_electrical["P_el Fan [kW]"].sum() + energy_system_results_electrical["P_el Heat Pump [kW]"].sum())

        autarky_rate_system_abs = energy_system_results_electrical["Autarky Rate [-]"].sum(
        ) / len(energy_system_results_electrical)

        # Copying Slices of data Frame for grid cover, Heat Pump electrical power, electrical storage output power
        p_el_heat_pump = energy_system_results_electrical["P_el Heat Pump [kW]"].copy(
        )
        p_el_storage_out = energy_system_results_electrical["P_el elec. Storage [kW]"].copy(
        )
        p_el_pv = energy_system_results_electrical["P_el SRT ac [kW]"].copy()

        # filter only output power of storage. -> energy that is used to power either household electricity or heat pump from storage
        p_el_storage_out[p_el_storage_out > 0] = 0

        # The share of grid cover for the heat pump is the residual load of the energy storage cover and pv energy supply, because the heat pump has the first priority when it comes to pv power supply
        p_el_heat_pump_share_grid_cover = p_el_heat_pump - p_el_pv + p_el_storage_out
        # If the residual load is negative, a surplus even after supplying the heat pump is available
        p_el_heat_pump_share_grid_cover[p_el_heat_pump_share_grid_cover < 0] = 0

        # Console Results Output

        # Nominal thermal capacity of EMC and Boiler (resulting from thermal peak load)
        print("\nBoiler cap.:\t\t\t\t", round(
            energy_system_results_thermal["q Boiler [kW th.]"].max(), 2), "kW th")

        if combined_heat_and_power == True:
            print("EMC cap.:\t\t\t\t\t", round(
                energy_system_results_thermal["q CHP excess [kW th.]"].max(), 2), "kW th")
        else:
            pass

        print("\nSystem Autarky Rate:\t\t", round(autarky_rate_system_abs, 3))
        print("JAZ ASHP:\t\t\t\t\t", round(jaz_ashp, 2))

        if combined_heat_and_power == True:
            sum_chp_p_el = energy_system_results_electrical["P_el CHP [kW el.]"].sum(
            )
            capacity_chp = components.iloc[4]["Capacity"]
            if capacity_chp == 0:  # in case capacity is zero the divison later would be by zero
                sum_chp_p_el = 0
                capacity_chp = 1
            # CHP electrical generation / CHP installed capacity electric
            chp_full_load_hours = sum_chp_p_el / capacity_chp
            print("CHP Full load hours:\t\t", round(
                chp_full_load_hours, 1), "h")
        else:
            pass

        print("\nCAPEX:\t\t\t\t\t\t", round(components.loc['Sum']["CAPEX [€]"], 0), "€",
              "\nOPEX fix:\t\t\t\t\t", round(
                  components.loc['Sum']["OPEX fix [€]"], 0), "€",
              "\nOPEX var:\t\t\t\t\t", round(components.loc['Sum']["OPEX var [€]"], 0), "€")
        print("Net Energy Costs:\t\t\t", round(
            electric_energy_costs.loc['Bilancy']["Costs/Revenue over runtime [€]"], 0)*(-1), "€")
        print("Overall Costs over lifetime:", round(overall_system_costs, 0),
              "€\n- per Annum:\t\t\t\t", round(overall_system_costs/n, 0), "€/a")
        print("\nCO2e Emissions over lifetime:{}".format(round(overall_system_co2_emissions, 0)),
              "kg\n- per Annum:\t\t\t\t", round(overall_system_co2_emissions/n, 0), "kg/a\n\n")
        print("\n\n-> Costs for Fuel bilanced in OPEX var. as fuel\n-> Net Energy Costs consider electric energy costs & revenues from PV & CHP\n-> Autarky Rate only relates to electrical Consumption\n")

        optimization_results = 0
        hyper_volume_indicator = 0
        spacing = 0

        return optimization_results, energy_system_results_electrical, energy_system_results_thermal, hyper_volume_indicator, spacing, components

    else:
        pass

    # if optimization_mode == True:
    #     energy_system_results_electrical= 0
    #     energy_system_results_thermal   = 0
    # else:
    #     optimization_results = 0
    #     hyper_volume_indicator=0
    #     spacing =0
    # return optimization_results, energy_system_results_electrical,energy_system_results_thermal, hyper_volume_indicator, spacing, components <- hier gibt es einen Fehler wenn im Optimode, funktioniert mit "Components" nur wenn NICHT Optimode
