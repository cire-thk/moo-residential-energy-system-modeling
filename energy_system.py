# -*- coding: utf-8 -*-
"""
- Created within the Framework of a masterthesis @
faculty of Process Engineering, Energy and Mechanical Systems 
of the Cologne University of Applied Sciences -
@author: marius bartkowski

Contact:
marius.bartkowski@magenta.de or
marius.bartkowski@smail.th-koeln.de
"""

# Libs
import sys
import pandas as pd
import numpy as np
import math
from parametrization import heatflux_temperature_regression


class electrical_vehicle_ems():

    def __init__(self, residual_load_electric, ev_df, ev_plug_t_min_1, ev_plug_t_min_2, e_electrical_vehicle_t_min_1, soc_ev_t_min_1):
        """
        #########################################################################################################
        Plug-in Value #     Description
        #########################################################################################################
        0             #     Car unplugged and is discharging till minimum safety factor
        1             #     Car plugged and is charging till quantity needed to cover demand (plus safety factor)
        2             #     Car is in vacation. Stand-by position (does nothing)
        3             #     Car plugged-in before vacation, and must be fully charged
        #########################################################################################################
        """

        '_______optimisation parameters______'
        # residual power of Solar Panels
        self.residual_load_electric = residual_load_electric
        self.plug_status_t_min_1 = ev_plug_t_min_1  # plug in status at current timestep
        self.plug_status_t_min_2 = ev_plug_t_min_2  # plug in status at next timestep
        # EV battery capacity at current timestep
        self.e_capacity_t_min_1 = e_electrical_vehicle_t_min_1
        self.soc_t_min_1 = soc_ev_t_min_1
        self.residual_load_electric_2 = self.residual_load_electric
        # initialize the charging power of the vehicle
        self.p_charge_electrical_vehicle = 0

        '_______ EV specs _______ '
        # max EV Battery charging power [kW]
        self.p_electrical_vehicle_max = ev_df['Battery Charging Power']
        # total EV Battery Capacity [kWh]
        self.e_capacity_electrical_vehicle_max = ev_df['Battery Capacity']
        # total energy consumed per day [kWh]
        self.e_consumption_electrical_vehicle = ev_df['Energy Consumption'] * \
            ev_df['daily traveled distance']

        # soc that must not be fallen under
        self.safety_factor = ev_df['Safety Factor']
        self.soc_after_charging = self.soc_t_min_1 + \
            (self.p_electrical_vehicle_max /
             self.e_capacity_electrical_vehicle_max)  # soc if charged with max. charging power

        # EV discharging power when unplugged
        self.p_discharge_electrical_vehicle = self.e_consumption_electrical_vehicle / \
            ev_df['Daily Commuting period']  # discharging power per timestep [kW]

        # EV daily needed SoC
        self.soc_daily_min = self.e_consumption_electrical_vehicle / \
            self.e_capacity_electrical_vehicle_max + self.safety_factor
        self.min_capacity = self.e_capacity_electrical_vehicle_max * \
            self.soc_daily_min  # minimum daily capacity in kWh

        # Calculating Car's avaible capacity to charge
        self.free_capacity = self.e_capacity_electrical_vehicle_max - self.e_capacity_t_min_1

        # Setting Car SoC after returning from vacation to minimal Battery capacity
        if self.plug_status_t_min_1 == 2 and self.plug_status_t_min_2 == 1:
            # set battery capacity to minimum
            self.e_capacity_t_min_2 = self.e_capacity_electrical_vehicle_max*self.safety_factor
            self.soc_t_min_1 = self.e_capacity_t_min_2 / \
                self.e_capacity_electrical_vehicle_max
            # residual does not change
            self.residual_load_electric_2 = self.residual_load_electric
            return

        # Car is plugged-in:
        if self.plug_status_t_min_1 == 1:

            # in case PV has excess energy always charge car if not fully charged
            if self.residual_load_electric > 0:

                # here check for car state of charge
                if self.soc_t_min_1 < 1:
                    # does the PV excess energy exceeds charging power of car?
                    if self.residual_load_electric > self.free_capacity:

                        # make sure that car does not charge in one time step more that the allowed charging power
                        if self.p_electrical_vehicle_max <= self.free_capacity:
                            # Charge car with EV charging power
                            self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = self.residual_load_electric - \
                                self.p_charge_electrical_vehicle

                        elif self.p_electrical_vehicle_max > self.free_capacity:
                            # charge car till max free capacity
                            self.p_charge_electrical_vehicle = self.free_capacity
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = self.residual_load_electric - \
                                self.p_charge_electrical_vehicle

                    # is the PV excess energy less the charging power?
                    elif self.residual_load_electric <= self.free_capacity:

                        if self.p_electrical_vehicle_max <= self.residual_load_electric:
                            # charge car only with max. allowed power
                            self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = self.residual_load_electric - \
                                self.p_charge_electrical_vehicle

                        elif self.p_electrical_vehicle_max > self.residual_load_electric:
                            # charge car with entire PV power
                            self.p_charge_electrical_vehicle = self.residual_load_electric
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = 0
                            # self.residual_load_electric = 0 #important here to set it to zero, so that in the next step it will be checked if required daily SOC is met for EV

                else:  # will be met if the battery is fully charged. do nothing
                    self.e_capacity_t_min_2 = self.e_capacity_t_min_1
                    self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                        self.e_capacity_electrical_vehicle_max
                    self.residual_load_electric_2 = self.residual_load_electric

            # PV does not have excess energy. Here we take needed energy and subtract it from residual
            elif self.residual_load_electric <= 0:

                if self.soc_t_min_1 < self.soc_daily_min:

                    self.needed_capacity = self.min_capacity - self.e_capacity_t_min_1

                    if self.needed_capacity < self.p_electrical_vehicle_max:
                        # charge battery till minimum daily required (does not exceed max. charging power of battery!)
                        self.p_charge_electrical_vehicle = self.needed_capacity
                        self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                            self.p_charge_electrical_vehicle
                        self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                            self.e_capacity_electrical_vehicle_max
                        # update residual load
                        self.residual_load_electric_2 = self.residual_load_electric - \
                            self.p_charge_electrical_vehicle

                    elif self.needed_capacity > self.p_electrical_vehicle_max:
                        # charge with maximum allowed power
                        self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                        self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                            self.p_charge_electrical_vehicle
                        self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                            self.e_capacity_electrical_vehicle_max
                        # update residual variables
                        self.residual_load_electric_2 = self.residual_load_electric - \
                            self.p_charge_electrical_vehicle

                else:
                    # nothing changes. do not charge from the net.
                    self.residual_load_electric_2 = self.residual_load_electric
                    self.e_capacity_t_min_2 = self.e_capacity_t_min_1
                    self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                        self.e_capacity_electrical_vehicle_max

        # Car is un_plugged => discharge till safety factor:
        elif self.plug_status_t_min_1 == 0:

            if self.soc_t_min_1 > self.safety_factor:

                self.e_capacity_t_min_2 = self.e_capacity_t_min_1 - \
                    self.p_discharge_electrical_vehicle
                self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                    self.e_capacity_electrical_vehicle_max
                # residual is not affected, since car is not home!
                self.residual_load_electric_2 = self.residual_load_electric

        # Plugg-in Status 3, means car getting ready for vacation and must be fully charged! Here make sure to fully charge from electrical net!
        # same logic as in state 1, only difference is that the maximum charge is 100% instead of daily minimum
        elif self.plug_status_t_min_1 == 3:
            # in case PV has excess energy always charge car if not fully charged

            if self.residual_load_electric > 0:

                # here check for car state of charge
                if self.soc_t_min_1 < 1:
                    # does the PV excess energy exceeds charging power of car?
                    if self.residual_load_electric > self.free_capacity:

                        # make sure that car does not charge in one time step more that the allowed charging power
                        if self.p_electrical_vehicle_max <= self.free_capacity:
                            # Charge car with EV charging power
                            self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = self.residual_load_electric - \
                                self.p_charge_electrical_vehicle

                        elif self.p_electrical_vehicle_max > self.free_capacity:
                            # charge car till max free capacity
                            self.p_charge_electrical_vehicle = self.free_capacity
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = self.residual_load_electric - \
                                self.p_charge_electrical_vehicle

                    # is the PV excess energy less the charging power?
                    elif self.residual_load_electric <= self.free_capacity:

                        if self.p_electrical_vehicle_max <= self.residual_load_electric:
                            # charge car only with max. allowed power
                            self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = self.residual_load_electric - \
                                self.p_charge_electrical_vehicle

                        elif self.p_electrical_vehicle_max > self.residual_load_electric:
                            # charge car with entire PV power
                            self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = -self.p_electrical_vehicle_max
                            # self.residual_load_electric = 0 #important here to set it to zero, so that in the next step it will be checked if required daily SOC is met for EV
                else:
                    # nothing changes. do not charge from the net. maximum allowed SoC is reached!
                    self.residual_load_electric_2 = self.residual_load_electric
                    self.e_capacity_t_min_2 = self.e_capacity_t_min_1
                    self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                        self.e_capacity_electrical_vehicle_max

            elif self.residual_load_electric <= 0:

                if self.soc_t_min_1 < 1:

                    self.needed_capacity = self.free_capacity

                    if self.needed_capacity < self.p_electrical_vehicle_max:
                        # charge battery till minimum daily required (does not exceed max. charging power of battery!)
                        self.p_charge_electrical_vehicle = self.needed_capacity
                        self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                            self.p_charge_electrical_vehicle
                        self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                            self.e_capacity_electrical_vehicle_max
                        # update residual load
                        self.residual_load_electric_2 = self.residual_load_electric - \
                            self.p_charge_electrical_vehicle

                    elif self.needed_capacity > self.p_electrical_vehicle_max:
                        # charge with maximum allowed power
                        self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                        self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                            self.p_charge_electrical_vehicle
                        self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                            self.e_capacity_electrical_vehicle_max
                        # update residual variables
                        self.residual_load_electric_2 = self.residual_load_electric - \
                            self.p_charge_electrical_vehicle

                else:
                    # nothing changes. do not charge from the net.
                    self.residual_load_electric_2 = self.residual_load_electric
                    self.e_capacity_t_min_2 = self.e_capacity_t_min_1
                    self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                        self.e_capacity_electrical_vehicle_max

        else:  # will actually be only met if the plug in state is 2
            self.e_capacity_t_min_2 = self.e_capacity_t_min_1
            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                self.e_capacity_electrical_vehicle_max
            self.residual_load_electric_2 = self.residual_load_electric
            pass

        # update capacity and soc
        self.soc_t_min_1 = self.soc_t_min_2
        self.e_capacity_t_min_1 = self.e_capacity_t_min_2


class electrical_vehicle_normal():

    def __init__(self, residual_load_electric, ev_df, ev_plug_t_min_1, ev_plug_t_min_2, e_electrical_vehicle_t_min_1, soc_ev_t_min_1):

        ##########################################################################################################
        # Plug-in Value #     Description
        ##########################################################################################################
        # 0             #     Car unplugged and is discharging till minimum safety factor
        # 1             #     Car plugged and will charge till maximum value
        # 2             #     Car is in vacation. Discharging according to vacation consumption
        ##########################################################################################################
        '_______optimisation parameters______'
        # residual power of Solar Panels
        self.residual_load_electric = residual_load_electric
        self.plug_status_t_min_1 = ev_plug_t_min_1  # plug in status at current timestep
        self.plug_status_t_min_2 = ev_plug_t_min_2  # plug in status at next timestep
        # EV battery capacity at current timestep
        self.e_capacity_t_min_1 = e_electrical_vehicle_t_min_1
        self.soc_t_min_1 = soc_ev_t_min_1
        self.residual_load_electric_2 = self.residual_load_electric
        # initialize the charging power of the vehicle
        self.p_charge_electrical_vehicle = 0

        '_______ EV specs _______ '
        # max EV Battery charging power [kW]
        self.p_electrical_vehicle_max = ev_df['Battery Charging Power']
        # total EV Battery Capacity [kWh]
        self.e_capacity_electrical_vehicle_max = ev_df['Battery Capacity']
        # total energy consumed per day [kWh]
        self.e_consumption_electrical_vehicle = ev_df['Energy Consumption'] * \
            ev_df['daily traveled distance']

        # soc that must not be fallen under
        self.safety_factor = ev_df['Safety Factor']

        # EV discharging power when unplugged
        self.p_discharge_electrical_vehicle = self.e_consumption_electrical_vehicle / \
            ev_df['Daily Commuting period']  # discharging power per timestep [kW]

        # EV discharging power when in vacation
        self.p_discharge_electrical_vehicle_vacation = self.p_discharge_electrical_vehicle * \
            2  # assuming first that on vacation the car consumes double the power

        # EV daily needed SoC
        self.soc_daily_min = self.e_consumption_electrical_vehicle / \
            self.e_capacity_electrical_vehicle_max + self.safety_factor
        self.min_capacity = self.e_capacity_electrical_vehicle_max * \
            self.soc_daily_min  # minimum daily capacity in kWh

        # Calculating Car's avaible capacity to charge
        self.free_capacity = self.e_capacity_electrical_vehicle_max - self.e_capacity_t_min_1

        # charging battery always till maximum power. First use PV residual power, then use the grid power if no residual was available
        if self.plug_status_t_min_1 == 1:

            # in case PV has excess energy always charge car if not fully charged
            if self.residual_load_electric > 0:

                # here check for car state of charge
                if self.soc_t_min_1 < 1:
                    # does the PV excess energy exceeds charging power of car?
                    if self.residual_load_electric > self.free_capacity:

                        # make sure that car does not charge in one time step more that the allowed charging power
                        if self.p_electrical_vehicle_max <= self.free_capacity:
                            # Charge car with EV charging power
                            self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = self.residual_load_electric - \
                                self.p_charge_electrical_vehicle

                        elif self.p_electrical_vehicle_max > self.free_capacity:
                            # charge car till max free capacity
                            self.p_charge_electrical_vehicle = self.free_capacity
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = self.residual_load_electric - \
                                self.p_charge_electrical_vehicle

                    # is the PV excess energy less the charging power?
                    elif self.residual_load_electric <= self.free_capacity:

                        if self.p_electrical_vehicle_max <= self.residual_load_electric:
                            # charge car only with max. allowed power
                            self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = self.residual_load_electric - \
                                self.p_charge_electrical_vehicle

                        elif self.p_electrical_vehicle_max > self.residual_load_electric:
                            # charge car with entire PV power
                            self.p_charge_electrical_vehicle = self.residual_load_electric
                            self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                                self.p_charge_electrical_vehicle
                            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                                self.e_capacity_electrical_vehicle_max
                            self.residual_load_electric_2 = 0
                            # self.residual_load_electric = 0 #important here to set it to zero, so that in the next step it will be checked if required daily SOC is met for EV

                else:  # will be met if the battery is fully charged. do nothing
                    self.e_capacity_t_min_2 = self.e_capacity_t_min_1
                    self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                        self.e_capacity_electrical_vehicle_max
                    self.residual_load_electric_2 = self.residual_load_electric

            # PV does not have excess energy. Here we take needed energy and subtract it from residual
            elif self.residual_load_electric <= 0:

                # if battery is not full, also charge it till maximum capacity
                if self.soc_t_min_1 < 1:

                    if self.free_capacity < self.p_electrical_vehicle_max:
                        # charge battery till free available capacity (does not exceed max. charging power of battery!)
                        self.p_charge_electrical_vehicle = self.free_capacity
                        self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                            self.p_charge_electrical_vehicle
                        self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                            self.e_capacity_electrical_vehicle_max
                        # update residual load
                        self.residual_load_electric_2 = self.residual_load_electric - \
                            self.p_charge_electrical_vehicle

                    elif self.free_capacity > self.p_electrical_vehicle_max:
                        # charge with maximum allowed power
                        self.p_charge_electrical_vehicle = self.p_electrical_vehicle_max
                        self.e_capacity_t_min_2 = self.e_capacity_t_min_1 + \
                            self.p_charge_electrical_vehicle
                        self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                            self.e_capacity_electrical_vehicle_max
                        # update residual variables
                        self.residual_load_electric_2 = self.residual_load_electric - \
                            self.p_charge_electrical_vehicle

                else:
                    # nothing changes. do not charge from the net.
                    self.residual_load_electric_2 = self.residual_load_electric
                    self.e_capacity_t_min_2 = self.e_capacity_t_min_1
                    self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                        self.e_capacity_electrical_vehicle_max

        # discharge according to power consumption during week day commuted distance
        elif self.plug_status_t_min_1 == 0:

            if self.soc_t_min_1 > self.safety_factor:

                self.e_capacity_t_min_2 = self.e_capacity_t_min_1 - \
                    self.p_discharge_electrical_vehicle
                self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                    self.e_capacity_electrical_vehicle_max
                # residual is not affected, since car is not home!
                self.residual_load_electric_2 = self.residual_load_electric

            else:  # dont change. this case is met, when SoC value falls below safety factor
                self.e_capacity_t_min_2 = self.e_capacity_t_min_1
                self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                    self.e_capacity_electrical_vehicle_max
                # residual is not affected, since car is not home!
                self.residual_load_electric_2 = self.residual_load_electric

        # discharge according to power consumption during vacations
        elif self.plug_status_t_min_1 == 2:

            if self.soc_t_min_1 > self.safety_factor:
                # print("discharging VACATION")
                self.e_capacity_t_min_2 = self.e_capacity_t_min_1 - \
                    self.p_discharge_electrical_vehicle_vacation
                self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                    self.e_capacity_electrical_vehicle_max
                # residual is not affected, since car is not home!
                self.residual_load_electric_2 = self.residual_load_electric

            else:  # dont change. this case is met, when SoC value falls below safety factor
                self.e_capacity_t_min_2 = self.e_capacity_t_min_1
                self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                    self.e_capacity_electrical_vehicle_max
                # residual is not affected, since car is not home!
                self.residual_load_electric_2 = self.residual_load_electric

        # will never be actually met
        else:
            self.e_capacity_t_min_2 = self.e_capacity_t_min_1
            self.soc_t_min_2 = self.e_capacity_t_min_2 / \
                self.e_capacity_electrical_vehicle_max
            self.residual_load_electric_2 = self.residual_load_electric
            pass

        # update capacity and soc
        self.soc_t_min_1 = self.soc_t_min_2
        self.e_capacity_t_min_1 = self.e_capacity_t_min_2


class solar_rooftiles():
    def __init__(self, timestep, ambient_temperature, irradiation, wind_speed, building, hybrid_solar_rooftiles, regression_coefficients):

        # Get Parameters
        self.timestep = timestep
        self.regression_coefficients = regression_coefficients
        self.ambient_temperature = ambient_temperature
        self.irradiation_west = irradiation["West"]["POA Global [W/m2]"][timestep]
        self.irradiation_east = irradiation["East"]["POA Global [W/m2]"][timestep]
        self.wind_speed_west = wind_speed["West"]["Wind Speed [m/s]"][timestep]
        self.wind_speed_east = wind_speed["East"]["Wind Speed [m/s]"][timestep]
        self.building = building
        self.hybrid_solar_rooftiles = hybrid_solar_rooftiles

        # this can be optimised by using dictionaries for local variables and loop the irradiation direction
        if self.irradiation_west >= 50:

            # PV Model electrical
            # Module temperature according to Faiman model
            self.t_module_west = self.ambient_temperature + self.irradiation_west / \
                (self.hybrid_solar_rooftiles["U'0"] +
                 self.hybrid_solar_rooftiles["U'1"] * self.wind_speed_west)

            # temperature correction method 1 according to IEC 60891:2022-10
            self.i_shortcircuit_tempraturecorrected = self.hybrid_solar_rooftiles["Impp"] + self.hybrid_solar_rooftiles["Ik"] * (
                (self.irradiation_west/self.hybrid_solar_rooftiles["Irradiation STC"])-1) + self.hybrid_solar_rooftiles["Temperature_coeff._alpha"]*(self.t_module_west - self.hybrid_solar_rooftiles["Temperature STC"])
            self.u_opencircuit_tempraturecorrected = self.hybrid_solar_rooftiles["Umpp"] - self.hybrid_solar_rooftiles["Series Resistance Rs"] * (
                self.i_shortcircuit_tempraturecorrected - self.hybrid_solar_rooftiles["Impp"]) + self.hybrid_solar_rooftiles["Temperature_coeff._betha"] * (self.t_module_west - self.hybrid_solar_rooftiles["Temperature STC"])

            self.eta_solar_roof_tile_west = (self.u_opencircuit_tempraturecorrected * self.i_shortcircuit_tempraturecorrected) / (
                self.irradiation_west * self.hybrid_solar_rooftiles["solar_cell_area"])

            if self.eta_solar_roof_tile_west < 0:
                self.eta_solar_roof_tile_west = 0

            self.p_solar_roof_tile_electrical_dc_west = (
                self.eta_solar_roof_tile_west * self.irradiation_west * self.building["area_solar_roof_tiles_west"]) / 1000
            self.p_solar_roof_tile_electrical_ac_west = self.p_solar_roof_tile_electrical_dc_west * \
                hybrid_solar_rooftiles["Inverter Efficiency"]

            self.air_mass_flow_output_temperature_west = self.t_module_west - (self.t_module_west * heatflux_temperature_regression(
                self.irradiation_west, regression_coefficients[0][0], regression_coefficients[0][1]))

        else:

            self.t_module_west = self.ambient_temperature
            self.air_mass_flow_output_temperature_west = self.ambient_temperature

            self.u_opencircuit_tempraturecorrected = 0
            self.i_shortcircuit_tempraturecorrected = 0
            self.eta_solar_roof_tile_west = 0

            self.p_solar_roof_tile_electrical_dc_west = 0
            self.p_solar_roof_tile_electrical_ac_west = 0

        if self.irradiation_east >= 50:

            # PV Model electrical
            self.t_module_east = self.ambient_temperature + self.irradiation_east / \
                (self.hybrid_solar_rooftiles["U'0"] +
                 self.hybrid_solar_rooftiles["U'1"] * self.wind_speed_east)

            self.i_shortcircuit_tempraturecorrected = self.hybrid_solar_rooftiles["Impp"] + self.hybrid_solar_rooftiles["Ik"] * (
                (self.irradiation_east/self.hybrid_solar_rooftiles["Irradiation STC"])-1) + self.hybrid_solar_rooftiles["Temperature_coeff._alpha"]*(self.t_module_east - self.hybrid_solar_rooftiles["Temperature STC"])
            self.u_opencircuit_tempraturecorrected = self.hybrid_solar_rooftiles["Umpp"] - self.hybrid_solar_rooftiles["Series Resistance Rs"] * (
                self.i_shortcircuit_tempraturecorrected - self.hybrid_solar_rooftiles["Impp"]) + self.hybrid_solar_rooftiles["Temperature_coeff._betha"] * (self.t_module_east - self.hybrid_solar_rooftiles["Temperature STC"])

            self.eta_solar_roof_tile_east = (self.u_opencircuit_tempraturecorrected * self.i_shortcircuit_tempraturecorrected) / (
                self.irradiation_east * self.hybrid_solar_rooftiles["solar_cell_area"])

            if self.eta_solar_roof_tile_east < 0:
                self.eta_solar_roof_tile_east = 0

            self.p_solar_roof_tile_electrical_dc_east = (
                self.eta_solar_roof_tile_east * self.irradiation_east * self.building["area_solar_roof_tiles_east"]) / 1000
            self.p_solar_roof_tile_electrical_ac_east = self.p_solar_roof_tile_electrical_dc_east * \
                hybrid_solar_rooftiles["Inverter Efficiency"]

            self.air_mass_flow_output_temperature_east = self.t_module_east - (self.t_module_east * heatflux_temperature_regression(
                self.irradiation_east, regression_coefficients[0][0], regression_coefficients[0][1]))

        else:

            self.t_module_east = self.ambient_temperature
            self.air_mass_flow_output_temperature_east = self.ambient_temperature

            self.u_opencircuit_tempraturecorrected = 0
            self.i_shortcircuit_tempraturecorrected = 0
            self.eta_solar_roof_tile_east = 0

            self.p_solar_roof_tile_electrical_dc_east = 0
            self.p_solar_roof_tile_electrical_ac_east = 0

        # Here the values for each roof orientation has to be united for the oncoming calculations
        # this variable is irrelevant for further calculation.
        self.t_module = self.t_module_west
        # This temperature will determine the efficiency of the air sourced heatpump since the heatpumps evaporator draws heat from the solar roof tiles air circulation
        # It is assumed that the solar roof tile ventilation system can choose wether the fresh air first travels through the east or the west side.
        self.air_mass_flow_output_temperature = self.air_mass_flow_output_temperature_west if self.air_mass_flow_output_temperature_west > self.air_mass_flow_output_temperature_east else self.air_mass_flow_output_temperature_east

        self.eta_solar_roof_tile = self.eta_solar_roof_tile_west

        self.p_solar_roof_tile_electrical_dc = self.p_solar_roof_tile_electrical_dc_west + \
            self.p_solar_roof_tile_electrical_dc_east
        self.p_solar_roof_tile_electrical_ac = self.p_solar_roof_tile_electrical_ac_west * \
            self.p_solar_roof_tile_electrical_ac_east


class electrical_energy_storage():
    """
        Variable NameError               Variable discriptions:

        residual_load_electric              power demand of entire system (+ve if PV-system has surplus power and represents excess power. 
                                           -ve means demand is higher the production)

        e_electrical_storage_t_min_1        electrical storage capacity at current timestep
        e_electrical_storage_t              electrical storage capacity at next timestep

        electrical_storage                  Dataframe containing all Battery Specs

        soc_t_min_1                         State of Charge at the current timestep 

        p_electrical_storage                Charging power at timestep
        """

    def __init__(self, residual_load_electric, e_electrical_storage_t_min_1, electrical_storage, soc_t_min_1):

        # Get Parameters
        self.residual_load_electric = residual_load_electric
        self.e_electrical_storage_t_min_1 = e_electrical_storage_t_min_1
        self.electrical_storage = electrical_storage
        self.soc_t_min_1 = soc_t_min_1

        # calculating the loss in battery capacity due to Standing losses
        self.electrical_storage_standing_losses = self.e_electrical_storage_t_min_1 * \
            electrical_storage["Standing losses"]

        # checking if battery still has capacity (kWh), and remove the standing losses
        if (self.e_electrical_storage_t_min_1 - self.electrical_storage_standing_losses) >= 0:
            # Substract standing losses if storage isn't empty already
            self.e_electrical_storage_t_min_1 -= self.electrical_storage_standing_losses
        else:
            pass

        # Estimating free storage capacity (that could be charged) for time step t
        self.free_storage_capacity_before = electrical_storage[
            "e_electrical_storage_usable"] - self.e_electrical_storage_t_min_1

        # CHARGING BATTERY
        # Positive residual load means PV surplus -> Charging storage
        if self.residual_load_electric > 0:
            # If residual load is higher than free capacity, charge to full capacity respecting losses due to charging (efficiency)
            if self.residual_load_electric >= self.free_storage_capacity_before/electrical_storage["Efficiency"]:

                # calculating power needed to charge the battery respecting power charging losses
                '''HERE DID NOT RESPECT MAX CHARGING POWER OF BATTERY!'''
                self.p_electrical_storage = self.free_storage_capacity_before / \
                    electrical_storage["Efficiency"]  # Charge storage fully

                self.e_electrical_storage_t = self.e_electrical_storage_t_min_1 + self.p_electrical_storage * \
                    electrical_storage["Efficiency"]  # Storage Capacity for time step t

            else:  # In case of not charging the storage to nominal capacity
                # Charging power = residual load respecting efficiency
                self.p_electrical_storage = self.residual_load_electric
                self.e_electrical_storage_t = self.e_electrical_storage_t_min_1 + \
                    self.p_electrical_storage * \
                    electrical_storage["Efficiency"]  # New storage capacity
            # Estimate SOC
            self.soc = self.soc_t_min_1 + (((self.p_electrical_storage*electrical_storage["Efficiency"])/(1*electrical_storage["e_electrical_storage_usable"])) - (
                (self.electrical_storage_standing_losses/(1*electrical_storage["e_electrical_storage_usable"]))))*100

        if self.residual_load_electric <= 0:  # Negative residual load -> decharging storage
            # /electrical_storage["Efficiency"]): # Decharge storage fully respecting efficiency, if the residual load is bigger than the available capacity.
            if self.residual_load_electric*-1 >= self.e_electrical_storage_t_min_1:
                self.e_electrical_storage_t = 0
                if self.e_electrical_storage_t_min_1 != 0:  # If storage has capacity left in t-1, fully decharge
                    self.p_electrical_storage = - self.e_electrical_storage_t_min_1 * \
                        electrical_storage["Efficiency"]
                else:  # If storage is empty, decharging power = 0
                    self.p_electrical_storage = 0
                # Estimate SOC. In this case, the efficiency is already respected
                self.soc = self.soc_t_min_1 + (((self.p_electrical_storage/electrical_storage["Efficiency"])/(1*electrical_storage["e_electrical_storage_usable"])) - (
                    (self.electrical_storage_standing_losses/(1*electrical_storage["e_electrical_storage_usable"]))))*100
                self.e_electrical_storage_t = self.e_electrical_storage_t_min_1 + \
                    self.p_electrical_storage/electrical_storage["Efficiency"]

            else:  # If the residual load is lower than available storage capacity
                # /electrical_storage["Efficiency"] # Storage covers residual load fully
                self.p_electrical_storage = self.residual_load_electric

                self.soc = self.soc_t_min_1 + (((self.p_electrical_storage/electrical_storage["Efficiency"])/(1*electrical_storage["e_electrical_storage_usable"])) - (
                    (self.electrical_storage_standing_losses/(1*electrical_storage["e_electrical_storage_usable"]))))*100

                self.e_electrical_storage_t = self.e_electrical_storage_t_min_1 + \
                    (self.p_electrical_storage /
                     electrical_storage["Efficiency"])
                # self.p_electrical_storage = self.residual_load_electric

        # Eliminate very small (unreasonable) values for soc
        if self.soc < 0.5:
            self.soc = 0
        else:
            pass

        # Eliminate very small (unreasonable) values for the storage level shortly below 0
        if self.e_electrical_storage_t < 0:
            self.e_electrical_storage_t = 0
        else:
            pass


class heat_pump_ashp():
    def __init__(self, t_in, heat_pump):

        self.heat_pump = heat_pump
        self.t_in = t_in
        self.t_out = heat_pump["sink temperature"]
        # Spaceholder for p_el Heat Pump [kW]. Changed after Warm Water Storage Thermal Energy Balance
        self.p_el = 0
        # Spaceholder for q_th Heat Pump [kW]. Changed after Warm Water Storage Thermal Energy Balance
        self.q_th = 0

        # COP Calculation for ASHP as in Ruhnau et.al.
        self.hp_cop = 6.08 - 0.09 * \
            (self.t_out - self.t_in) + 0.0005 * (self.t_out - self.t_in)**2
        self.hp_qth_max = heat_pump["q_nom"]
        self.hp_pel_max = heat_pump["q_nom"] / self.hp_cop


class warm_water_storage_mixed():
    def __init__(self, system_ops, heat_pump_energies, ww_storage, heat_pump, cp_w, combined_heat_and_power, chp_specs, e_ww_storage_t_min_1, building_demand, fan_heat_pump):
        # This is a simplified model for a warm water storage with 3 nodes of equal volume. The main aspect of the model is to fulfill the first principle of thermodynamics.
        # The storage multiple in- and outflows and two heat exchangers for hot tap water and the air source heat pump. the surface area of the heat exchangers scales linearly with the volume of the storage which is a decision variable.
        # This part of the model has great potential to be optimised towards more realistic modelling.
        # There are multi-node models which can improve this models accuracy
        self.e_demand_heating_th = system_ops.e_demand_heating_th
        self.e_demand_ww_th = system_ops.e_demand_ww_th
        self.system_ops = system_ops

        # Extracted thermal energy with regards to nominal energy level
        self.e_ww_storage_diff_nom = ww_storage["Q_storage_nom"] - \
            e_ww_storage_t_min_1
        self.q_boiler_th = 0
        "Warm_Water_Storage_Temperature_and_Heat_Exchanger_Calculation"

        "____Storage Temperatures at t_____"
        self.t_ww_storage_lower = ww_storage["t_upper"] - (self.e_ww_storage_diff_nom / (
            ww_storage["Watermass"] * cp_w * ww_storage["load_factor_f1"] * (1 / 3600)))  # bottom temperature of storage
        # this estimation of the bottom temperature is inaccurate. the applied equation expresses which temperature difference the storage temperature of a storage with fully mixed temperature nodes undergoes if it delivers a certain amount of heat (self.e_ww_storage_diff_nom)
        # self.e_ww_storage_diff_nom can never be lower than the difference between the max energy level and the minimum allowed energy level since after every timestep the energy level has to atleast meet the minimum allowed energy level

        # Mean Storage temperature or the storage temperature in the middle node of the storage
        self.t_ww_storage_mean = (
            self.t_ww_storage_lower + ww_storage["t_upper"]) / 2

        # Assumption: heating circuit feed temperature = mean temperature due to its location in the middle of the storage
        self.heating_circuit_feed_t = self.t_ww_storage_mean

        "_____Heating circuit_____"

        # If the heating circle outlet temperature lies above the bottom storage temperature, the return flow increase is bypassed
        if self.t_ww_storage_lower < ww_storage["t_heating_outlet"]:
            return_flow_increase = False
        else:
            return_flow_increase = True     # in general trying to keep the bottom nodes temperature at a minimum is beneficient for solar thermal heat generation and in this case heat pumps. even though there is a set sink temperature for the heat pump in this model, this feature is included
            # if the return flow is bypassed the whole space heating demand is met by the boiler alone and the storage is not supplying any heat to the space heating circle
            # anyways this should never be the case with how self.t_ww_storage_lower is modeled
            pass

        # if the heating circle inlet temperature is > set temperature of heating circle (f.e. 55 °C), the mass flow can be reduced for providing the same amount of energy for space heating
        if self.heating_circuit_feed_t >= ww_storage["t_heating_inlet"] and return_flow_increase == True:
            self.heating_circuit_mass_flow = self.e_demand_heating_th / \
                (cp_w * (self.heating_circuit_feed_t -
                 ww_storage["t_heating_outlet"]))
            self.e_storage_heating_out_th = self.e_demand_heating_th
            self.e_reheat_heating_circle_th = 0
        # In case that the HC inlet temperature is below the desired temperature. In this case the heating water has to be reheated by the boiler
        elif return_flow_increase == True:
            # heating circle mass flow for set inlet and outlet temperatures
            self.heating_circuit_mass_flow = self.e_demand_heating_th / \
                (cp_w * (ww_storage["t_heating_inlet"] -
                 ww_storage["t_heating_outlet"]))
            # the heating circle inlet temperature is below the set temperature of the heating circle inlet. the residual energy demand to reach the the temperature is provided by reheating through an external heat generator (f.e. boiler).
            self.e_storage_heating_out_th = self.heating_circuit_mass_flow * \
                cp_w * (self.heating_circuit_feed_t -
                        ww_storage["t_heating_outlet"])
            # energy demand for reheating the inlet temperature on the set heating inlet temperature
            self.e_reheat_heating_circle_th = self.e_demand_heating_th - \
                self.e_storage_heating_out_th
        # If the Return Flow increase is bypassed, the Heating Circle is operated by the boiler only
        else:
            # Therefore the heat energy leaving the storage is zero, due to bypassing the return flow increase mechanism
            self.e_storage_heating_out_th = 0
            # If the boiler covers the heat demand alone, there is nothing left for the boiler to reheat
            self.e_reheat_heating_circle_th = self.e_demand_heating_th

        if self.e_demand_heating_th > 0:
            # If Heating demand has to be met, Circulation Pump for HC is activated
            p_circulation_pump_hc = ww_storage["P Circulation Pump"]
        else:
            p_circulation_pump_hc = 0

        "_____Warm Water Heat Exchanger_____"
        # If the heating system is operating, the bottom temperature is increased due to the feedback of the heating system return into the storage (return raise)
        if self.e_demand_heating_th > 0:
            # the bottom temperature than is asumed as the arithmethic mean of the heating system return temperature and the bottom temperature
            self.t_ww_storage_lower = (
                self.t_ww_storage_lower + ww_storage["t_heating_outlet"]) / 2
            # this case can atleast yield a temperature of
        else:
            pass

        self.t_log_warm_water_he = ((self.t_ww_storage_lower - ww_storage["t_cw"]) - (ww_storage["t_upper"] - ww_storage["t_ww"])) / np.log(
            (self.t_ww_storage_lower - ww_storage["t_cw"])/(ww_storage["t_upper"] - ww_storage["t_ww"]))  # Logharithmic temperature difference heat exchanger - counter current.
        # self.t_log_warm_water_he = ((ww_storage["t_upper"] - ww_storage["t_ww"]) - (self.t_ww_storage_lower - ww_storage["t_cw"]))/ np.log((ww_storage["t_upper"] - ww_storage["t_ww"])/(self.t_ww_storage_lower - ww_storage["t_cw"])) # direct current
        self.q_warm_water_he = (ww_storage["u_heat_exchanger"] * ww_storage["warm_water_he_area"]
                                * self.t_log_warm_water_he) / 1000  # max. Heat Exchanger power in dependency of t_log

        if self.q_warm_water_he < self.e_demand_ww_th:  # if the energy amount of warm water exceeds the power of the warm water heat exchanger, the amount of energy which has to be added to the storage by reheating has to be found in order to ensure the warm water supply

            # required logharithmic temperature difference for the heat exchager to ensure the warm water supply
            self.t_log_req = (self.e_demand_ww_th * 1000) / \
                (ww_storage["u_heat_exchanger"] *
                 ww_storage["warm_water_he_area"])

            self.t_ww_lower_min = self.t_ww_storage_lower
            self.t_log_warm_water_he_iter = self.t_log_warm_water_he

            # iterativly finding required storage bottom temperature for a t_log that ensures warm water supply
            while self.t_log_warm_water_he_iter < self.t_log_req and self.t_ww_lower_min < ww_storage["t_upper"]:
                self.t_log_warm_water_he = ((self.t_ww_lower_min - ww_storage["t_cw"]) - (ww_storage["t_upper"] - ww_storage["t_ww"])) / np.log(
                    (self.t_ww_lower_min - ww_storage["t_cw"])/(ww_storage["t_upper"] - ww_storage["t_ww"]))  # Logharithmic temperature difference heat exchanger - counter current.
                # self.t_log_warm_water_he = ((ww_storage["t_upper"] - ww_storage["t_ww"]) - (self.t_ww_lower_min - ww_storage["t_cw"]))/ np.log((ww_storage["t_upper"] - ww_storage["t_ww"])/(self.t_ww_lower_min - ww_storage["t_cw"])) # direct current
                # thermal energy amount to add to the storage that warm water supply is secured
                self.e_ww_storage_ww_reheat = ww_storage["Watermass"] * cp_w * (
                    self.t_ww_lower_min - self.t_ww_storage_lower) * (1 / 3600)
                # If storage has to be reheated, Circulation Pump for RH is activated
                p_circulation_pump_rh = ww_storage["P Circulation Pump"]
                self.t_ww_lower_min += 0.5  # increase temperature in 0.5 K steps
        else:
            self.e_ww_storage_ww_reheat = 0
            p_circulation_pump_rh = 0

        "_____Solar Circle Heat Exchanger_____"
        self.e_ww_storage_withdrawal = e_ww_storage_t_min_1 - \
            self.e_storage_heating_out_th - \
            self.e_demand_ww_th - ww_storage["q_storage_loss"]

        self.t_ww_storage_lower_t = ww_storage["t_upper"] - (
            (ww_storage["Q_storage_nom"] - self.e_ww_storage_withdrawal) / (ww_storage["Watermass"] * cp_w * (1 / 3600)))

        # If the heating system is operating, the bottom temperature is increased due to the feedback of the heating system return into the storage (return raise)
        if self.e_demand_heating_th > 0:
            # the bottom temperature than is asumed as the arithmethic mean of the heating system return temperature and the bottom temperature
            self.t_ww_storage_lower_t = (
                self.t_ww_storage_lower_t + ww_storage["t_heating_outlet"]) / 2
        else:
            pass

        # Mean Storage temperature
        self.t_ww_storage_mean_t = (
            self.t_ww_storage_lower_t + ww_storage["t_upper"])/2

        # If the mean storage temperature would fall under the sensor temperature, the sensor temperature is set for the calculation for the power of the solar-circle heat exchanger. It is assumed that a reheating by the solar circle is taking place when the mean storage temperature would sink below the sensor temperature
        if self.t_ww_storage_mean_t < ww_storage["t_sensor"]:
            self.t_ww_storage_mean_t = ww_storage["t_sensor"]
        else:
            pass

        # minimum bottom temperature according to the sensor temperature
        self.t_ww_storage_lower_min = 2 * \
            ww_storage["t_sensor"] - ww_storage["t_upper"]
        if self.t_ww_storage_lower_t < self.t_ww_storage_lower_min:  # The same applies for the bottom temperature
            self.t_ww_storage_lower_t = self.t_ww_storage_lower_min
        else:
            pass

        # self.t_log_solar_circle_he_upper_fraction_part = (heat_pump["t_hp_outlet"] - self.t_ww_storage_lower_t) - (heat_pump["t_hp_inlet"] - self.t_ww_storage_mean_t) #old
        self.t_log_solar_circle_he_upper_fraction_part = (heat_pump["t_hp_inlet"] - self.t_ww_storage_lower_t) - (
            heat_pump["t_hp_outlet"] - self.t_ww_storage_mean_t)  # counter current
        # self.t_log_solar_circle_he_ln_content = (heat_pump["t_hp_outlet"] - self.t_ww_storage_lower_t)/(heat_pump["t_hp_inlet"] - self.t_ww_storage_mean_t) #old
        self.t_log_solar_circle_he_ln_content = (heat_pump["t_hp_inlet"] - self.t_ww_storage_lower_t)/(
            heat_pump["t_hp_outlet"] - self.t_ww_storage_mean_t)  # counter current
        if self.t_log_solar_circle_he_ln_content <= 0:  # in this case t_log will be zero
            self.t_log_solar_circle_he_upper_fraction_part = 0
            self.t_log_solar_circle_he_ln_content = 2
        # Logharithmic temperature difference heat exchanger - counter current.
        self.t_log_solar_circle_he = self.t_log_solar_circle_he_upper_fraction_part / \
            np.log(self.t_log_solar_circle_he_ln_content)

        self.q_solar_circle_he = (ww_storage["u_heat_exchanger"] * ww_storage["solar_circle_he_area"]
                                  * self.t_log_solar_circle_he) / 1000  # max. Heat Exchanger power in dependency of t_log

        "____Storage at t__"
        # The energy demand which has to be supplied by the warm water storage and eventually additional heating components is the space heating demand covered by the storage, warm water demand, necessary reheated energy amount for securing warm water demand and q storage loss
        self.e_demand_th_total = self.e_storage_heating_out_th + self.e_demand_ww_th + \
            self.e_ww_storage_ww_reheat + ww_storage["q_storage_loss"]
        # the reaheating for the heating cycle that is done by the boiler takes away from the energy the warm water storage has to deliver. Hence this value can be taken out of the balance "What enters and what leaves the storage?"
        self.q_storage_th_out = -1 * \
            (self.e_demand_th_total - 0*self.e_reheat_heating_circle_th)
        # Storage level set for t set on storage level t-1 for beginning of time step t
        self.e_ww_storage_t = e_ww_storage_t_min_1 + self.q_storage_th_out

        # Power of circulation pumps is the sum of both pumps (if active)
        self.p_circulation_pumps = p_circulation_pump_rh + p_circulation_pump_hc

        "____Heat_Pump_Operation____"
        # The Heat Pumps thermal energy input is limited by the heat exchanger of the solar circle
        if heat_pump_energies.hp_qth_max > self.q_solar_circle_he:
            heat_pump_energies.hp_qth_max = self.q_solar_circle_he
            heat_pump_energies.hp_pel_max = heat_pump_energies.hp_qth_max / \
                heat_pump_energies.hp_cop

        # Surplus Solar energy for Heat Pump
        self.p_pv_pre_surplus = system_ops.p_pv - \
            (building_demand["Sum Electricity [kWh]"] +
             self.p_circulation_pumps + fan_heat_pump["p_el"]/1000)
        # self.heat_pump["p_nom"]: # The powering of the heat pump by PV energy has the highest priority after supplying the general electricity demand within the energy system. -> General Electricity > Pumps and Fans > Heat Pump Electricity > Storage Electricity. The PV supply after covering general electricity (huoseholds) is used as surplus energy to preheat the storage
        if self.p_pv_pre_surplus > heat_pump_energies.hp_pel_max:
            # self.heat_pump["p_nom"] # If the PV energy after covering general electricity demand exceeds the nominal power of the heat pump, the pv energy used to power the heat pump is p_nom (heat Pump)
            self.p_pv_hp = heat_pump_energies.hp_pel_max
        else:
            # Otherwise, the surplus pv energy applied for the heat pump is the residual of pv energy and household electricity demand
            self.p_pv_hp = self.p_pv_pre_surplus
            if self.p_pv_hp < 0:    # If pv energy is 0, p_pv_hp can get potentially negative. -> Zeroing
                self.p_pv_hp = 0

        # The surplus thermal energy is derived by the COP at t
        self.q_hp_th_surplus = self.p_pv_hp * heat_pump_energies.hp_cop

        # the surplus thermal energy by the heat pump Also has to be limited according to the heat exchanger power at timestep t
        if self.q_hp_th_surplus > heat_pump_energies.hp_qth_max:
            self.q_hp_th_surplus = heat_pump_energies.hp_qth_max
        else:
            pass

        "___Warm Water Storage Operations___"
        # General

        # Reasses Operating mode due to adjusted energy demand
        # If the storage level falls below the set temperature (set storage level) the heat pump is activated
        if self.system_ops.hp == 1 and (self.e_ww_storage_t >= ww_storage["q_storage_on"]):
            self.system_ops.hp = 0
            self.system_ops.fan = 0

        # If the storage level falls below the set temperature (set storage level) the heat pump is activated
        if self.system_ops.boiler == 1 and (self.e_ww_storage_t >= ww_storage["q_storage_on"]):
            self.system_ops.boiler = 0

        # print('HP:', self.system_ops.hp, 'BO:', self.system_ops.boiler, 'e w', self.e_ww_storage_t, 'e on', ww_storage["q_storage_on"])
        # Operating Modes
        # Case: Energy Demand is totally met by Warm Water Storage, no need for Heat Pump or Boiler Operation
        if bool(self.system_ops.boiler) == False and bool(self.system_ops.hp) == False:
            # -> self.e_ww_storage_t stays as determined in system ops
            # surplus pv energy is available which can be used to preheat the storage by the heat pump.
            if self.q_hp_th_surplus > 0:

                # If the surplus is larger as the available space in the storage, it is limited to the available capacity
                if self.q_hp_th_surplus > (ww_storage["Q_storage_nom"] - self.e_ww_storage_t):
                    heat_pump_energies.q_th = ww_storage["Q_storage_nom"] - \
                        self.e_ww_storage_t
                    heat_pump_energies.p_el = heat_pump_energies.q_th / heat_pump_energies.hp_cop
                else:
                    heat_pump_energies.q_th = self.q_hp_th_surplus
                    heat_pump_energies.p_el = heat_pump_energies.q_th / heat_pump_energies.hp_cop

                self.q_storage_th_in = heat_pump_energies.q_th
                self.e_ww_storage_t = e_ww_storage_t_min_1 + \
                    self.q_storage_th_out + self.q_storage_th_in
                self.ops = "HP Surplus"

            else:   # no surplus energy equals only covering thermal energy demand by storage capacity
                self.q_storage_th_in = 0
                heat_pump_energies.q_th = 0
                heat_pump_energies.p_el = 0
                self.ops = "None"
        else:
            pass
        "___Initialize CHP Variables___"
        self.p_el_chp = 0
        self.q_th_chp = 0
        self.p_el_emc = 0
        self.q_th_emc = 0
        self.eta_el_chp = 0
        self.eta_th_chp = 0
        self.fe_kg_chp = 0
        "___Heat_Pump_Operation_Mode____ "
        if bool(self.system_ops.hp) == True:
            # print("q_on", ww_storage["q_storage_on"], "eww", self.e_ww_storage_t, "ww_rh", self.e_ww_storage_ww_reheat)
            self.ops = "HP"
            # Warm Water Storage Balance without additional heating capacity of h2 boiler needed.

            # Case HP1: Thermal energy demand can be fully met by heat pump and storage capacity. No surplus energy by HP
            if self.e_ww_storage_t < ww_storage["q_storage_on"]:
                # When Storage Capacity is below q_on (Predefined switch on point), its recharged until q_on is reached. IF surplus PV energy is available, its used to power the heat pump
                self.q_storage_th_in = ww_storage["q_storage_on"] - \
                    self.e_ww_storage_t

                # If the the energy difference between storage level and set level exceeds the maximum thermal energy of the heat pump, the maximum thermal energy of the heat pump is feed into the storage
                if self.q_storage_th_in >= heat_pump_energies.hp_qth_max:
                    self.q_storage_th_in = heat_pump_energies.hp_qth_max
                    # The heat pump thermal energy is the maximum heat pump thermal energy at timestep t
                    heat_pump_energies.q_th = heat_pump_energies.hp_qth_max
                    heat_pump_energies.p_el = heat_pump_energies.hp_pel_max
                else:
                    heat_pump_energies.q_th = self.q_storage_th_in
                    heat_pump_energies.p_el = heat_pump_energies.q_th / heat_pump_energies.hp_cop

                    if self.q_hp_th_surplus > 0:
                        # If the surplus is larger as the available space in the storage, it is limited to the available capacity
                        if self.q_hp_th_surplus > (ww_storage["Q_storage_nom"] - self.e_ww_storage_t):
                            heat_pump_energies.q_th = ww_storage["Q_storage_nom"] - \
                                self.e_ww_storage_t
                            heat_pump_energies.p_el = heat_pump_energies.q_th / heat_pump_energies.hp_cop
                            self.q_storage_th_in = heat_pump_energies.q_th
                        else:
                            if self.q_hp_th_surplus > heat_pump_energies.q_th:
                                heat_pump_energies.q_th = self.q_hp_th_surplus
                                heat_pump_energies.p_el = heat_pump_energies.q_th / heat_pump_energies.hp_cop
                                self.q_storage_th_in = heat_pump_energies.q_th
                                self.ops += " + HP Surplus"
                            else:
                                heat_pump_energies.q_th  # Stays the same
                                self.q_storage_th_in = heat_pump_energies.q_th
                                # self.q_storage_th_in -> stays the same

                    else:
                        pass

                self.e_ww_storage_t = e_ww_storage_t_min_1 + \
                    self.q_storage_th_out + self.q_storage_th_in
                # If CHP is in the system configuration, PV-electricity generation can not cover the demand from the HP and the storage has still capapcity open, the CHP system will operate
                if combined_heat_and_power == True and heat_pump_energies.q_th > self.q_hp_th_surplus and self.e_ww_storage_t < ww_storage["Q_storage_nom"] and self.p_pv_hp < heat_pump_energies.p_el:
                    self.p_el_chp = heat_pump_energies.p_el - self.p_pv_hp
                    self.chp_energies = fuel_cell_chp(heat_pump_energies,
                                                      ww_storage, system_ops,
                                                      chp_specs,
                                                      self.p_el_chp,
                                                      self.e_ww_storage_t
                                                      )
                    self.q_th_chp = self.chp_energies.q_th
                    self.p_el_chp = self.chp_energies.p_el
                    self.p_el_emc = self.chp_energies.p_el_emc
                    self.q_th_emc = self.chp_energies.q_th_emc
                    self.eta_el_chp = self.chp_energies.eta_el
                    self.eta_th_chp = self.chp_energies.eta_th
                    self.fe_kg_chp = self.chp_energies.fe_kg
                    self.q_storage_th_in += self.q_th_chp
                    self.ops += self.chp_energies.ops
                else:
                    pass

            else:
                pass

            self.e_ww_storage_t = e_ww_storage_t_min_1 + \
                self.q_storage_th_out + self.q_storage_th_in

            # Renewed query for the storage level after the operation of the heat pump in time step t. If e_ww_storage is still below the set level, it means that the heat pump even on nominal power can not reheat the storage to the set level. The boiler is activated to reheat the storage to the set level
            if self.e_ww_storage_t < ww_storage["q_storage_on"]:
                # the heat demand of the boiler in this case is the residual load between e_ww_storage after heat pump and q_on_storage
                self.q_boiler_th += ww_storage["q_storage_on"] - \
                    self.e_ww_storage_t
                # Since the boiler energy can already contains the space heating demand only the deficit of the warm water storage is added
                self.q_storage_th_in += ww_storage["q_storage_on"] - \
                    self.e_ww_storage_t
                # the storage level after the reheating by the boiler is calcualte again
                self.e_ww_storage_t = e_ww_storage_t_min_1 + \
                    self.q_storage_th_out + self.q_storage_th_in
                self.ops += " + Boiler Support"

            else:
                pass
        else:
            pass

        "___H2_Boiler_Operation_Mode____ "
        if bool(system_ops.boiler) == True:

            # Warm Water Storage Balance only with h2 boiler (when t ambient < bivalency point)
            # self.e_ww_storage_t < ww_storage["q_storage_on"]:                                          # Case:

            # When Storage Capacity is below q_on (Predefined switch on point), its recharged until q_on is reached
            self.q_boiler_th = ww_storage["q_storage_on"] - self.e_ww_storage_t
            self.q_storage_th_in = self.q_boiler_th

            heat_pump_energies.q_th = 0
            self.e_ww_storage_t = e_ww_storage_t_min_1 - self.e_demand_th_total + \
                self.q_storage_th_in  # += self.q_storage_th_in
            self.ops = "Boiler (Reheat)"
        else:
            pass

        if self.e_reheat_heating_circle_th > 0:
            # Reheating of the heating circle inlet temperature if it is required. In this case, an external heat generator (f.e. boiler) has to be used to increase the temperature. This form of reheating does not effect the storage directly
            self.q_boiler_th += self.e_reheat_heating_circle_th
            self.ops += ' + Reheating Heating circle'
            system_ops.boiler = 1
        else:
            pass

        # self.ops == 'HP Surplus' or self.ops == 'HP Surplus + CHP' or self.ops == 'HP Surplus + EMC + CHP'
        if self.ops == 'HP Surplus' or self.ops == 'HP Surplus + CHP' or self.ops == 'HP Surplus + EMC + CHP':
            system_ops.hp = 1
            system_ops.fan = 1
        # self.ops == 'HP + EMC + CHP + Boiler Support' or self.ops == 'HP + CHP + Boiler Support' or
        elif self.ops == 'HP + EMC + CHP + Boiler Support' or self.ops == 'HP + CHP + Boiler Support' or self.ops == 'HP + Boiler Support':
            system_ops.boiler = 1
        else:
            pass

        if return_flow_increase == False:  # Add Heat Demand to total energy demand for balancing purposes, even if it is covered by the boiler completely
            self.e_demand_th_total += self.e_demand_heating_th
        else:
            pass

        if self.q_boiler_th > 0:
            system_ops.boiler = 1
        else:
            pass


class h2_boiler():
    def __init__(self, ww_storage_balance, system_ops, boiler_specs):
        self.ww_storage_balance = ww_storage_balance
        self.system_ops = system_ops
        self.boiler_specs = boiler_specs

        self.q_boiler_th = self.ww_storage_balance.q_boiler_th
        self.p_boiler_el = self.boiler_specs["p_el"]*self.system_ops.boiler
        # Boiler fuel demand [kg]
        self.fe_boiler = (
            self.q_boiler_th / self.boiler_specs["efficiency"]) / self.boiler_specs["Ho"]


class fuel_cell_chp():
    def __init__(self, heat_pump_energies, ww_storage, system_ops, chp_specs, p_el, e_ww_storage_t):

        self.p_el = p_el
        self.ops = ""
        # Reference values for Fuel Cell System electric efficiency curve
        self.eta_el_ref = [0, 55, 60, 37]
        # Reference values for Fuel Cell normed electric power
        self.p_el_ref_normed = [0, 0.020, 0.067, 1]

        # If the power of the heat pump exeeds the nominal power of the CHP, p_el is set to p_el_nom
        if self.p_el > chp_specs["p_el_nom_chp"]:
            self.p_el = chp_specs["p_el_nom_chp"]
        else:
            pass

        # Normed electrical power of chp
        self.p_el_normed = self.p_el/chp_specs["p_el_nom_chp"]
        # interpolating Fuel Cell System efficiency according to fuel cell system power
        self.eta_el = np.interp(
            self.p_el_normed, self.p_el_ref_normed, self.eta_el_ref) / 100

        # Avoid nan or -inf vlaues by deviding through zero
        if (self.p_el == 0) or (self.eta_el == 0):
            self.fe_kwh = 0
            self.fe_kg = 0
        else:
            # Calculating fuel consumption of CHP in kWh H2
            self.fe_kwh = self.p_el / self.eta_el
            # Convert Fuel consumption to kg H2
            self.fe_kg = self.fe_kwh / chp_specs["Hu Hydrogen"]

        # Thermal Efficiency [-]
        self.eta_th = (1 - self.eta_el) * \
            chp_specs["eta_hrs"] * chp_specs["zeta_heat_loss"]
        self.eta_ges = self.eta_th + self.eta_el  # overall efficiency of CHP
        self.q_th = self.fe_kwh * self.eta_th  # Thermal power output of CHP

        # If the efficiency is below 20 %  due to part-load operation, the electric power output is set to 0 to avoid extreme fuel consumptions.
        if self.eta_el < 0.20:
            self.p_el = 0
            self.q_th = 0  # Therefore, the heat production of the chp is also zero
            self.q_th_emc = 0  # Therefore, there can't be excess thermal energy due to the CHP
            self.p_el_emc = 0
            system_ops.chp = 0  # Therefore, the CHP is also not running
        else:

            # Free Capacity
            self.q_ww_storage_free_capacity = ww_storage["Q_storage_nom"] - \
                e_ww_storage_t
            # In case thermal energy of the CHP exceeds the free capacity of the thermal storage
            if self.q_th > self.q_ww_storage_free_capacity:
                # Excess thermal energy as the difference between CHP heat production and free storage capacity
                self.q_th_emc = self.q_th - self.q_ww_storage_free_capacity
                # All thermal energy of CHP is stored until thermal storage capacity limit
                self.q_th = self.q_ww_storage_free_capacity
                self.p_el_emc = self.q_th_emc / \
                    chp_specs["EER EMC"]            # [kW]
                system_ops.emc = 1
                self.ops += " + EMC"  # Emergency cooling unit
            else:
                self.q_th_emc = 0
                self.p_el_emc = 0
                system_ops.emc = 0
                # self.q_th -> Stays the same if the thermal energy of the CHP does not exceed the free storage capacity

            system_ops.chp = 1
        if system_ops.chp == 1:
            self.ops += " + CHP"


class energy_system_operating_strategy():
    def __init__(self, irradiation, air_mass_flow_output_temperature, e_ww_storage_t_min_1, e_electrical_storage_t_min_1, p_pv, building_demand, ww_storage, heat_pump, combined_heat_and_power):

        self.e_demand_heating_th = building_demand["Sum Space Heating Net [kWh]"]
        self.e_demand_ww_th = building_demand["Sum Warm Water [kWh]"]
        # total thermal energy demand at timestep t is: heating demand, warm water demand and warm water storage losses
        self.e_demand_th_total = self.e_demand_heating_th + \
            self.e_demand_ww_th + ww_storage["q_storage_loss"]
        # e_ww_storage_t describes the warm water storage level after supplying the energy demand. The systems operating strategy sets according to that.
        self.e_ww_storage_t = e_ww_storage_t_min_1 - self.e_demand_th_total
        self.p_pv = p_pv

        if self.e_ww_storage_t < 0:
            self.e_ww_storage_t = 0

        # Heat Pump
        # If the outside temperature is below the bivalency point, the boiler is supplying the energy demand
        if air_mass_flow_output_temperature < heat_pump["Bivalency Point"]:
            self.hp = 0
        # If the storage level falls below the set temperature (set storage level) the heat pump is activated
        elif self.e_ww_storage_t < ww_storage["q_storage_on"]:
            self.hp = 1
        elif self.e_ww_storage_t < self.e_demand_th_total:  # ?
            self.hp = 1
        else:
            self.hp = 0  # If the storage level is above the set level after supply, no reheating is neccessary

        # Fan of HP
        # If the irradiation is above 50 W/m2 or the heat pump is running, the fan is activated
        if irradiation >= 50 or self.hp == True:
            self.fan = 1
        else:
            self.fan = 0

        # H2-Boiler
        # If the ambient temperature is below the bivalency point and the storage level is below the set level, the boiler is activated
        if air_mass_flow_output_temperature < heat_pump["Bivalency Point"] and self.e_ww_storage_t < ww_storage["q_storage_on"]:
            self.boiler = 1
        # if the storage level is below the energy demand, the boiler is also activated
        elif air_mass_flow_output_temperature < heat_pump["Bivalency Point"] and self.e_ww_storage_t < self.e_demand_th_total:
            self.boiler = 1
        else:
            self.boiler = 0

        # H2-CHP Fuel Cell
        # Emergency cooling Unit is integrated in CHP class
        if combined_heat_and_power == True:
            if self.hp == 1:
                self.chp = 1
                # Initiation of emergency cooling variable
                self.emc = 0
            else:
                self.chp = 0
                # Initiation of emergency cooling variable
                self.emc = 0
        else:
            pass


# %%from electrical_vehicle import ev_connection_profile
# from electrical_vehicle import ev_connection_profile
# from energy_system import electrical_vehicle

# ev_df = {
#     'Battery Capacity': 22,
#     'Battery Charging Power': 5,
#     'Energy Consumption':0.158,
#     'Daily Commuting period':20 ,
#     "Plug In": ev_connection_profile()
#     }
# # ev_df['Battery Capacity'] = 22                      #EV Battery Capacity in kWh
# # ev_df['Battery Charging Power'] = 100               #EV Battery Charging power kW
# # ev_df['Energy Consumption'] = 0.158                 #consumption of the EV per traveled distance (kWh/km)
# # ev_df['Daily Commuting']= 20                        #average covered distance in a week day Monday till Friday (km)
# # ev_df['Daily Commuting period'] = 10                # daily unplugged time of car (h) used to calculate the discharging power of the car
# # ev_df["Plug In"] = ev_connection_profile()          #plug-in state of the vehicle, showing when the car is avaible for charging
# e_electrical_vehicle_t_min_1 = ev_df['Battery Capacity']*0.5
# soc_ev_t_min_1 = e_electrical_vehicle_t_min_1/ ev_df['Battery Capacity']


# # print(ev_df['Plug In'])

# ev_storage_balance = electrical_vehicle(residual_load_electric= -10,
#                                         ev_df=ev_df,
#                                         ev_plug_t_min_1=ev_df["Plug In"][142],
#                                         ev_plug_t_min_2=ev_df["Plug In"][143],
#                                         e_electrical_vehicle_t_min_1=e_electrical_vehicle_t_min_1,
#                                         soc_ev_t_min_1=soc_ev_t_min_1)
# print("SoC_daily_min: {}".format(ev_storage_balance.soc_daily_min))
# # print("SoC_after_p_max: {}".format(ev_storage_balance.soc_after_charging))
# # print(ev_storage_balance.min_capacity)
# print("final capacity: {}".format(ev_storage_balance.e_capacity_t_min_1))
# print("SoC_2: {}".format(ev_storage_balance.soc_t_min_1))
# print("residuald: {}".format(ev_storage_balance.residual_load_electric_2))
# %%
