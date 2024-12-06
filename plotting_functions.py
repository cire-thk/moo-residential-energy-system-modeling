

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pymoo.visualization.pcp import PCP
from pymoo.mcdm.pseudo_weights import PseudoWeights
import matplotlib as mpl
import matplotlib.pyplot as plt

# Path & Data Handling
import os

# import glob
import datetime as dt
import pandas as pd
import numpy as np

# Importing Multi-Optimization Function
from optimization import optimize_multi_objective

path = os.getcwd()


def plot_Pareto_Fronts(optimization_mode, opt_results, optimal_solutions_df, sys_config, optimization_specs):
    """This function creates the Pareto Fronts Plot, where the plotted points are distinguished according to the simulated cases.
           Here the distuished cases are the used fuel type, car EMS system (with/withoun), and the type of the simulated renovation case.
           This gives more visualised insight about what are the combinations that are producing the ParetoFronts"""

    if optimization_mode == True:

        fuel_type = []
        ev_ems_type = []
        renovation_type = []
        for i in optimal_solutions_df["Type of Boiler Fuel"]:
            if i <= 1:
                fuel_type.append("Natural gas")
            elif i <= 2:
                fuel_type.append("H2 gas")

        for i in optimal_solutions_df["EV-EMS Case"]:
            if i <= 1:
                ev_ems_type.append("no EMS")
            elif i <= 2:
                ev_ems_type.append("EMS")

        for i in optimal_solutions_df["Renovation Case"]:
            if i < 1:
                renovation_type.append("no renovation")
            elif i < 2:
                renovation_type.append("roof")
            elif i < 3:
                renovation_type.append("roof, windows")
            elif i < 4:
                renovation_type.append("roof, windows, walls")
            elif i < 5:
                renovation_type.append("roof, windows, walls, basement")
            elif i <= 6:
                renovation_type.append("full renovation")

        # Define the categories and their mappings
        fuel_types = ["Natural gas",
                      "H2 gas"]
        ev_ems_types = ["no EMS",
                        "EMS"]
        renovation_types = [
            "no renovation",
            "roof",
            "roof, windows",
            "roof, windows, walls",
            "roof, windows, walls, basement",
            "full renovation"
        ]

        # Initialize dictionaries to hold the results
        x_results = {}
        y_results = {}

        # Populate the dictionaries using nested loops
        for fuel in fuel_types:
            for ems in ev_ems_types:
                for idx, reno in enumerate(renovation_types):
                    key = f"{fuel}_{ems}_reno{idx}"
                    try:
                        x_results[key] = [opt_results[0][i] for i in range(len(
                            fuel_type)) if fuel_type[i] == fuel and ev_ems_type[i] == ems and renovation_type[i] == reno]
                        y_results[key] = [opt_results[1][i] for i in range(len(
                            fuel_type)) if fuel_type[i] == fuel and ev_ems_type[i] == ems and renovation_type[i] == reno]
                    except:
                        x_results[key] = [opt_results["Annuity Cost"][i] for i in range(len(
                            fuel_type)) if fuel_type[i] == fuel and ev_ems_type[i] == ems and renovation_type[i] == reno]
                        y_results[key] = [opt_results["CO2 emissions"][i] for i in range(len(
                            fuel_type)) if fuel_type[i] == fuel and ev_ems_type[i] == ems and renovation_type[i] == reno]

        # plotting conbined results
        fig, ax = plt.subplots(figsize=(15, 12))

        for fuel in fuel_types:
            for ems in ev_ems_types:
                for idx, reno in enumerate(renovation_types):
                    key = f"{fuel}_{ems}_reno{idx}"
                    if len(x_results[key]) > 0:
                        if fuel == "Natural gas":
                            marker_fuel = "o"
                        else:
                            marker_fuel = "^"

                        if ems == "no EMS":

                            marker_ems = "black"
                        else:
                            marker_ems = "purple"

                        if idx == 0:
                            marker_reno = "red"
                        elif idx == 1:
                            marker_reno = "orange"
                        elif idx == 2:
                            marker_reno = "violet"
                        elif idx == 3:
                            marker_reno = "yellow"
                        elif idx == 4:
                            marker_reno = "blue"
                        elif idx == 5:
                            marker_reno = "cyan"

                        plt.plot(x_results[key], y_results[key], markersize=15, markeredgewidth=3, linestyle='', marker=marker_fuel,
                                 markeredgecolor=marker_ems, color=marker_reno, label=f"Gas:[{fuel}]  Car:[{ems}]  Renovation:[{reno}]")

        plt.ylabel("CO$_{2}$ Emissions per Year [kg/a]", fontsize=25)
        plt.xlabel("Annuity of Total System Costs  [€/a]", fontsize=25)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.grid()

        # create folder for Results
        folder_name = os.path.join(
            path+"\Output\Optimizations\Illustrations", dt.date.today().strftime('%Y-%m-%d'))
        try:
            # create the folder first if it does not exist
            os.makedirs(folder_name)

            plt.savefig(folder_name + '\\ParetoFronts{}_{}_Gen{}_Pop{}_{}.eps'.format(sys_config,
                        optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], dt.date.today()), dpi=300, format='eps')
            plt.savefig(folder_name + '\\ParetoFronts{}_{}_Gen{}_Pop{}_{}.png'.format(sys_config,
                        optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], dt.date.today()), dpi=300, format='png')
        except:
            plt.savefig(folder_name + '\\ParetoFronts{}_{}_Gen{}_Pop{}_{}.eps'.format(sys_config,
                        optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], dt.date.today()), dpi=300, format='eps')
            plt.savefig(folder_name + '\\ParetoFronts{}_{}_Gen{}_Pop{}_{}.png'.format(sys_config,
                        optimization_specs["Algorithm"], optimization_specs["Generations"], optimization_specs["Population Size"], dt.date.today()), dpi=300, format='png')


def sensivity_analysis(lower_boundary, Sensitivity_step, upper_boundary, sensitivity_variable, specs_eco, optimization_mode, optimization_specs, combined_heat_and_power, max_pv_power, temp_data, energy_system_simulation, own_design, n):

    Sensitivity_results = pd.DataFrame(columns=[
                                       "optimization results", "CO2 emissions", "Annuity Cost", f"{sensitivity_variable}"])
    Sensitivity_variable = []
    Sensitivity_results_electricity_co2 = []
    Sensitivity_results_electricity_annuity = []
    optimization_results_Sensitivity = []
    hyper_volume = []
    spacing_results = []
    hold_old_price = specs_eco[f"{sensitivity_variable}"]
    for price in np.arange(lower_boundary, upper_boundary, Sensitivity_step):
        specs_eco[f"{sensitivity_variable}"] = price

        optimization_results, energy_system_results_electrical, energy_system_results_thermal, hyper_volume_indicator, spacing = optimize_multi_objective(optimization_mode=optimization_mode,
                                                                                                                                                          optimization_specs=optimization_specs,
                                                                                                                                                          combined_heat_and_power=combined_heat_and_power,
                                                                                                                                                          max_pv_power=max_pv_power,
                                                                                                                                                          temp_data=temp_data,
                                                                                                                                                          energy_system_simulation=energy_system_simulation,
                                                                                                                                                          own_design=own_design,
                                                                                                                                                          n=n)
        Sensitivity_variable.append(price)
        Sensitivity_results_electricity_annuity.append(
            optimization_results.F.T[0])
        Sensitivity_results_electricity_co2.append(optimization_results.F.T[1])
        optimization_results_Sensitivity.append(optimization_results)
        hyper_volume.append(hyper_volume_indicator)
        spacing_results.append(spacing)
    Sensitivity_results["optimization results"] = optimization_results_Sensitivity
    Sensitivity_results["CO2 emissions"] = Sensitivity_results_electricity_co2
    Sensitivity_results["Annuity Cost"] = Sensitivity_results_electricity_annuity
    Sensitivity_results[f"{sensitivity_variable}"] = Sensitivity_variable
    Sensitivity_results["hyper volume"] = hyper_volume
    Sensitivity_results["spacing"] = spacing_results

    # return old price to original value
    specs_eco[f"{sensitivity_variable}"] = hold_old_price
    return Sensitivity_results


def plotting_Sensitivity(ref, Sensitivity_results, Sensitivity_test):

    def parse_array(s):
        if isinstance(s, np.ndarray):
            return s
        elif isinstance(s, str):
            return np.array([float(x) for x in s.strip('[]').split()])
        else:
            raise ValueError("Input must be a string or numpy array")

    df = pd.DataFrame(
        columns=["CO2 emissions", "Annuity Cost", "{} price".format(Sensitivity_test)])
    # df= pd.DataFrame()
    df['CO2 emissions'] = Sensitivity_results['CO2 emissions']
    df['Annuity Cost'] = Sensitivity_results['Annuity Cost']

    df['CO2 emissions'] = df['CO2 emissions'].apply(parse_array)
    df['Annuity Cost'] = df['Annuity Cost'].apply(parse_array)

    for i in Sensitivity_results.index:

        ii = Sensitivity_results["CO2 emissions"][i]

        if isinstance(ii, str):
            # changes list of string to list of floating numbers
            emissions_float = [float(x) for x in ii.strip("[]").split()]
            z = [1]*len(emissions_float)
            result = [
                x * Sensitivity_results['{} price'.format(Sensitivity_test)][i] for x in z]
            df['{} price'.format(Sensitivity_test)][i] = result
            df['{} price'.format(Sensitivity_test)][i] = np.array(
                df['{} price'.format(Sensitivity_test)][i])

        else:
            z = [1]*len(Sensitivity_results["CO2 emissions"][i])
            result = [
                x * Sensitivity_results['{} price'.format(Sensitivity_test)][i] for x in z]
            df['{} price'.format(Sensitivity_test)][i] = result
            df['{} price'.format(Sensitivity_test)][i] = np.array(
                df['{} price'.format(Sensitivity_test)][i])

    df['CO2 emissions'] = df['CO2 emissions'].apply(parse_array)
    df['Annuity Cost'] = df['Annuity Cost'].apply(parse_array)

    # Convert 'electricity price' column values to lists of floats
    df['{} price'.format(Sensitivity_test)] = df['{} price'.format(
        Sensitivity_test)].apply(parse_array)

    # Erstellen des 3D-Plots
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection='3d')

    # reference line to be plotted
    line_x = np.array(ref["Annuity Cost"].values/1000)
    line_y = np.array(ref['{} price'.format(Sensitivity_test)])
    line_z = np.array(ref["CO2 emissions"].values/1000)

    # Sort the line_x and line_z to ensure proper connection order (optional step)
    sorted_indices = np.argsort(line_x)
    line_x = line_x[sorted_indices]
    line_z = line_z[sorted_indices]

    ax.plot(line_x, line_y, line_z, linewidth=5, color='black', zorder=5)

    # rescale
    df['Annuity Cost'] = df['Annuity Cost']/1000

    x = np.concatenate(df['Annuity Cost'].values)
    y = np.concatenate(df['{} price'.format(Sensitivity_test)].values)
    z = np.concatenate(df['CO2 emissions'].values/1000)  # in t/a

    # creating plot
    mappable = ax.plot_trisurf(x, y, z, cmap='RdYlGn_r')

    # Create colorbar with adjusted font size for the label
    colorbar = plt.colorbar(mappable=mappable, shrink=0.5)
    colorbar.set_label("CO$_2$ emissions [t/a]", fontsize=20, labelpad=30)

    # Beschriftung der Achsen
    ax.set_xlabel(
        'Annuity Cost [10$^{3}$ €$\cdot a^{-1}$]', fontsize=20, labelpad=40)

    if Sensitivity_test == "electricity":
        ax.set_ylabel('electricity price [€/kWh]', fontsize=20, labelpad=30)
        ax.set_yticks(np.arange(0.2, 0.5, 0.05))

    elif Sensitivity_test == "h2":
        ax.set_ylabel('H$_{2}$ price [€/kg]', fontsize=20, labelpad=30)
        ax.set_yticks(np.arange(0, 15, 3))
    else:
        ax.set_ylabel('NG price [€/kg]', fontsize=20, labelpad=30)
        ax.set_yticks(np.arange(1, 3.5, 0.5))

    # setting scale
    ax.set_xticks(np.arange(10, 30, 5))
    # adjusting x- and y-axis number sizes
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    # # plt.rc('ztick', labelsize=30)

    ax.legend()

    # Titel hinzufügen
   # plt.title('Change of CO2 emmisions and Annuity prices according to change in {} prices'.format(Sensitivity_test), fontsize=30)

    # create folder for Results
    folder_name = os.path.join(
        path+"\Output\Optimizations\Illustrations", dt.date.today().strftime('%Y-%m-%d'))

    try:
        # create the folder first if it does not exist
        os.makedirs(folder_name)
        # save
        plt.savefig(
            folder_name + '\\Sensitivity3D_{}.eps'.format(Sensitivity_test), dpi=300, format='eps')
        plt.savefig(
            folder_name + '\\Sensitivity3D_{}.png'.format(Sensitivity_test), dpi=300, format='png')
    except:
        plt.savefig(
            folder_name + '\\Sensitivity3D_{}.eps'.format(Sensitivity_test), dpi=300, format='eps')
        plt.savefig(
            folder_name + '\\Sensitivity3D_{}.png'.format(Sensitivity_test), dpi=300, format='png')


def Sensitivity_data_prep(Sensitivity_results, test):

    for i in Sensitivity_results.index:
        cleaned_str = Sensitivity_results["CO2 emissions"][i].strip('[]')
        # Split the string into individual number strings
        number_strings = cleaned_str.split()
        # Convert the individual number strings to floats
        number_list = [float(num) for num in number_strings]
        Sensitivity_results["CO2 emissions"][i] = np.array(number_list)

        cleaned_str = Sensitivity_results["Annuity Cost"][i].strip('[]')
        # Split the string into individual number strings
        number_strings = cleaned_str.split()
        # Convert the individual number strings to floats
        number_list = [float(num) for num in number_strings]
        Sensitivity_results["Annuity Cost"][i] = np.array(number_list)

        Sensitivity_results[f"{test} price"][i] = float(
            Sensitivity_results[f"{test} price"][i])
        Sensitivity_results["hyper volume"][i] = float(
            Sensitivity_results["hyper volume"][i])
        Sensitivity_results["spacing"][i] = float(
            Sensitivity_results["spacing"][i])

    return Sensitivity_results


def plot_PCP(ResultPath, ParetoFrontPath_Opt_Sol, ParetoFrontPath):
    "Plots Parallel Coordinate Plots (PCP), which is used to show all parameter combinations of the optimal solutions obtained in ParetoFronts"

    result_path = ResultPath  # Create Result Path

    ParetoFronts_Opt_Solutions = pd.read_excel(ParetoFrontPath_Opt_Sol)
    ParetoFronts_Results = pd.read_excel(ParetoFrontPath)

    # rounding up values for Gas type, EMS, and Renovation
    ParetoFronts_Opt_Solutions.iloc[:, 7] = np.ceil(
        ParetoFronts_Opt_Solutions.iloc[:, 7])
    ParetoFronts_Opt_Solutions.iloc[:, 8] = np.ceil(
        ParetoFronts_Opt_Solutions.iloc[:, 8])
    ParetoFronts_Opt_Solutions.iloc[:, 9] = np.floor(
        ParetoFronts_Opt_Solutions.iloc[:, 9])
    final_results_pcp = np.array([ParetoFronts_Opt_Solutions.iloc[:, 1],
                                  ParetoFronts_Opt_Solutions.iloc[:, 2],
                                  ParetoFronts_Opt_Solutions.iloc[:, 1],
                                  ParetoFronts_Opt_Solutions.iloc[:, 4],
                                  ParetoFronts_Opt_Solutions.iloc[:, 5],
                                  ParetoFronts_Opt_Solutions.iloc[:, 6],
                                  ParetoFronts_Opt_Solutions.iloc[:, 7],
                                  ParetoFronts_Opt_Solutions.iloc[:, 8],
                                  ParetoFronts_Opt_Solutions.iloc[:, 9],
                                  ParetoFronts_Results.iloc[:, 2],
                                  ParetoFronts_Results.iloc[:, 1]]
                                 )
    final_results_tradeoff = np.array(
        [ParetoFronts_Results.iloc[:, 1], ParetoFronts_Results.iloc[:, 2]])

    # MCDM Trade-Off Point for PF Solutions
    weights = np.array([0.5, 0.5])

    # Transpose
    final_results_tradeoff = np.transpose(final_results_tradeoff)
    final_results_pcp = np.transpose(final_results_pcp)

    final_results_tradeoff, pseudo_weights_2023_sys_a_tradeoff = PseudoWeights(
        weights).do(final_results_tradeoff, return_pseudo_weights=True)

    # Parrallel coordinate plots (PCPs)
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.serif'] = ' DejaVu Sans'
    plt.rcParams.update({'font.size': 13})

    pcp_plot_a_2023 = PCP(cmap=cm.cividis, figsize=(20, 10), labels=["P$_{SRT_{East}}$\n[kWp$_{el}$]",
                                                                     "P$_{SRT_{West}}$\n[kWp$_{el}$]",
                                                                     "E$_{BESS}$\n[kWh$_{el}$]", "V$_{TES}$\n[L]",
                                                                     "Q$_{ASHP_{nom}}$\n[kW$_{th}$]",
                                                                     "T$_{biv}$\n[°C]",
                                                                     "Boiler Fuel",
                                                                     "EMS",
                                                                     "Renovation",
                                                                     "CO$_{2}$\n[kg$\cdot$a$^{-1}$]",
                                                                     "C$_{a}$\n[kg$\cdot$a$^{-1}$]"], legend='check', tight_layout=True)

    pcp_plot_a_2023.set_axis_style(color="dimgrey")  # , alpha=0.5)

    # Set global font size for axis labels and tick labels
    plt.rcParams.update({'xtick.labelsize': 20,  # Font size for x-axis labels
                         'ytick.labelsize': 14,  # Font size for y-axis labels
                         'font.size': 14})  # Font size for other text

    pcp_plot_a_2023.add(final_results_pcp, alpha=0.3)  # alpha=0.3
    pcp_plot_a_2023.add(
        final_results_pcp[1], alpha=0.9, color="darkred", linewidth=2.5, label="min. CO$_{2}$")
    pcp_plot_a_2023.add(
        final_results_pcp[0], alpha=0.9, color='olivedrab', linewidth=2.5, label="min. C$_{a}$")
    pcp_plot_a_2023.add(final_results_pcp[final_results_tradeoff], linestyle='-.',
                        alpha=0.5, color='darkcyan', linewidth=3, label="eq. w. Trade-Off")

    pcp_plot_a_2023.show()

    plt.savefig(
        result_path + '\\PCP_Final_Results_{}.eps'.format(dt.date.today()), dpi=300, format='eps')
    plt.savefig(
        result_path + '\\PCP_Final_Results_{}.png'.format(dt.date.today()), dpi=300, format='png')
