'''
This file will generate the electrical yearly demand for charging an electrical vehicle (EV) Battery.




'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ev_ems_connection_profile(year=2013,
                        arrival_time=16,
                        departure_time=7,
                        vacation_departure_time= 9,
                        vacation_arrival_time= 22,
                        number_vacation_weekends = 8):
    
    # ################################################################################################################################
    # Variables                   #             Variable descriptions
    # ################################################################################################################################
    # ev_battery_capacity         #   Capacity of the EV's battery (kWh)
    # ev_energy_consumption       #   consumption of the EV per traveled distance (kWh/km)
    # daily_commuting             #   average covered distance in a week day Monday till Friday (km)
    # number_vacation_days        #   number of vacation days per year (Vacation here is considered only on weekends)
    # arrival_time                #   time when the EV is plugged in the house
    # departure_time              #   time when the EV is un-plugged from the house      
    # number_vacation_weekends    #   number of weekends in a year, where vacation is taken
    # year                        #   time step year
    # vacation_departure_time     #   time when the vacation starts on Saturday (time before unplugging car from charger)
    # vacation_arrival_time       #   time when the EV returns from vacation on Sunday (plug-in time)   
    # safety_factor               #   represents the State of Charge (%) of battery after returning from vacation 0->100
    # ################################################################################################################################
    
    
    
    # Creating Timestamp
    start = "01-01-{} 00:00".format(str(year))
    end = "31-12-{} 23:00".format(str(year))
    timestamp = pd.date_range(start=start, end=end, freq='H') # tz='UTC'
    
    ev_profile= pd.DataFrame({'Date': timestamp})

    #the days of the week in the function dt.dayofweek are represented accordingly
    #0: Monay   1:Tuesday   2:Wednesday     3:Thursday     4:Friday   5:Saturday    6:Sunday
    ev_profile['Week Day'] = ev_profile['Date'].dt.dayofweek
    
    #creating a plugged-in profile for the Electrical Vehicle (EV) under following assumptions
    # 0: car is unplugged    (used later for discharging battery)
    # 1: car is plugged-in   (used later for charging battery)
    # 2: other               (used for neither charging or discharging: while on vacation)
    
    for n in ev_profile.index:
        
        # during the weekdays the car is plugged-in before departure and at arrival till midnight
        if ( ev_profile.loc[n,'Week Day'] == 0 or 
             ev_profile.loc[n,'Week Day'] == 1 or 
             ev_profile.loc[n,'Week Day'] == 2 or 
             ev_profile.loc[n,'Week Day'] == 3 or
             ev_profile.loc[n,'Week Day'] == 4
        ):
            if ev_profile.loc[n,'Date'].hour < departure_time:
                
                ev_profile.loc[n,'Plug'] = 1
                
            elif ev_profile.loc[n,'Date'].hour >= departure_time and ev_profile.loc[n,'Date'].hour < arrival_time:
                
                ev_profile.loc[n,'Plug'] = 0
            
            else:
                
                ev_profile.loc[n,'Plug'] = 1
        
        #on weekend the car is always plugged-in unless there is a vacation
        else:
            ev_profile.loc[n,'Plug'] = 1
            
    
    #calculate step to reach vacation week
    step = int(8760/number_vacation_weekends)
    weekly_hours = 7*24  #12 hours additional to include the disregarded weekend hours in case first day in step is already in vacation
    
  
    for n in range(0,8760,step):
        
        
        #resetting weekly timestep looper to zero
        t_week = 0 #loops through the entire week, when there is a vacation
        while t_week < weekly_hours:
            
            
            #to prevent double vacations in one week
            if t_week == 0:
                if (ev_profile.loc[n+t_week,'Week Day'] == 5 or ev_profile.loc[n+t_week,'Week Day'] == 6):
                    t_week += 48  #skipping 2 days just to make sure that the weekend is avoided, in case step starts at weekend
                elif ev_profile.loc[n+t_week,'Week Day'] == 4 and ev_profile.loc[n+t_week,'Date'].hour >= arrival_time:
                    t_week += 48  #skipping 2 days just to make sure that the weekend is avoided, in case step starts at friday where condition is met => use weekend after that
            
            
            else:
                #making sure that loop does not exceed maximum number of yearly timesteps
                if n+ t_week >= 8759:
                    break
                #setting arrival time on friday before vacation to plugg-in Status=3; indicating that car here must be charged with maximum power
                elif ev_profile.loc[n+t_week,'Week Day'] == 4 and ev_profile.loc[n+t_week,'Date'].hour >= arrival_time:
                    
                    while not(ev_profile.loc[n+t_week,'Date'].hour == vacation_departure_time and ev_profile.loc[n+t_week,'Week Day'] == 5):
                        if n+t_week == 8759:
                            break
                        #just before vacation the car must be fully charged! thus giving plugg-in value a different value! 
                        ev_profile.loc[n+t_week,'Plug'] = 3
                        t_week += 1
                    ev_profile.loc[n+t_week,'Plug'] = 3
                    
                #while on vacation the plug-in status is different, because energy consumption of car on vacation is different than consumption during weekdays
                elif  ev_profile.loc[n+t_week,'Week Day'] == 5 and ev_profile.loc[n+t_week,'Date'].hour >= vacation_departure_time:
                    
                    
                    while not(ev_profile.loc[n+t_week,'Date'].hour == vacation_arrival_time and ev_profile.loc[n+t_week,'Week Day'] == 6):
                        
                        
                        if n+t_week == 8759:
                            break
                        ev_profile.loc[n+t_week,'Plug'] = 2
                        t_week += 1
            
            t_week += 1    
        
    return ev_profile

      
def ev_normal_connection_profile(year=2013,
                        arrival_time=16,
                        departure_time=7,
                        vacation_departure_time= 9,
                        vacation_arrival_time= 22,
                        number_vacation_weekends = 11):
    
    # Creating Timestamp
    start = "01-01-{} 00:00".format(str(year))
    end = "31-12-{} 23:00".format(str(year))
    timestamp = pd.date_range(start=start, end=end, freq='H') # tz='UTC'
    
    ev_profile= pd.DataFrame({'Date': timestamp})

    #the days of the week in the function dt.dayofweek are represented accordingly
    #0: Monay   1:Tuesday   2:Wednesday     3:Thursday     4:Friday   5:Saturday    6:Sunday
    ev_profile['Week Day'] = ev_profile['Date'].dt.dayofweek
    
    #creating a plugged-in profile for the Electrical Vehicle (EV) under following assumptions
    # 0: car is unplugged    (used later for discharging battery)
    # 1: car is plugged-in   (used later for charging battery
    for n in ev_profile.index:
        
        # during the weekdays the car is plugged-in before departure and at arrival till midnight
        if ( ev_profile.loc[n,'Week Day'] == 0 or 
             ev_profile.loc[n,'Week Day'] == 1 or 
             ev_profile.loc[n,'Week Day'] == 2 or 
             ev_profile.loc[n,'Week Day'] == 3 or
             ev_profile.loc[n,'Week Day'] == 4
        ):
            if ev_profile.loc[n,'Date'].hour < departure_time:
                
                ev_profile.loc[n,'Plug'] = 1
                
            elif ev_profile.loc[n,'Date'].hour >= departure_time and ev_profile.loc[n,'Date'].hour < arrival_time:
                
                ev_profile.loc[n,'Plug'] = 0
            
            else:
                
                ev_profile.loc[n,'Plug'] = 1
        
        #on weekend the car is always plugged-in unless there is a vacation
        else:
            ev_profile.loc[n,'Plug'] = 1
            
        #calculate step to reach vacation week
    step = int(8760/number_vacation_weekends)
    weekly_hours = 7*24 +12
    
    for n in range(0,8760,step):
                 
            #resetting weekly timestep looper to zero
            t_week = 0 #loops through the entire week, when there is a vacation
            while t_week < weekly_hours:
                
                #to prevent double vacations in one week
                if t_week == 0:
                    if (ev_profile.loc[n+t_week,'Week Day'] == 5 or ev_profile.loc[n+t_week,'Week Day'] == 6):
                        t_week += 48  #skipping 2 days just to make sure that the weekend is avoided, in case step starts at weekend
                    elif ev_profile.loc[n+t_week,'Week Day'] == 4 and ev_profile.loc[n+t_week,'Date'].hour >= arrival_time:
                        t_week += 12  #skipping 2 days just to make sure that the weekend is avoided, in case step starts at friday where condition is met => use weekend after that


                else:
                    #making sure that loop does not exceed maximum number of yearly timesteps
                    if n+t_week >= 8759:
                        break

                    #while on vacation the plug-in status is different, because energy consumption of car on vacation is different than consumption during weekdays
                    elif  ev_profile.loc[n+t_week,'Week Day'] == 5 and ev_profile.loc[n+t_week,'Date'].hour >= vacation_departure_time:
                        
                        while not(ev_profile.loc[n+t_week,'Date'].hour == vacation_arrival_time and ev_profile.loc[n+t_week,'Week Day'] == 6):
                            
                            if n+t_week == 8759:
                                break
                            ev_profile.loc[n+t_week,'Plug'] = 2
                            t_week += 1
                
                t_week += 1    
    
    return ev_profile

df = pd.DataFrame()
# df = ev_demand_generator()
df = ev_normal_connection_profile()
df.plot(x='Date', y='Plug')
plt.show()