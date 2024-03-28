import matplotlib.pyplot as plt
import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp  
        self.ki = ki  
        self.kd = kd 
        self.setpoint = setpoint  
        self.previous_error = 0  
        self.integral = 0  

    def update(self, current_temperature, time_elapsed):
        error = self.setpoint - current_temperature
        P = self.kp * error
        self.integral += error * time_elapsed
        I = self.ki * self.integral
        derivative = (error - self.previous_error) / time_elapsed
        D = self.kd * derivative
        self.previous_error = error
        output = P + I + D
        output = max(0, output)
        return output

class WaterHeaterWithPID:
    def __init__(self, initial_temperature, max_temperature, ambient_temperature,
                 surface_area, heat_transfer_coefficient, pid_controller):
        self.current_temperature = initial_temperature  
        self.max_temperature = max_temperature 
        self.ambient_temperature = ambient_temperature  
        self.surface_area = surface_area  
        self.heat_transfer_coefficient = heat_transfer_coefficient  
        self.pid_controller = pid_controller  

    def heat_water(self, time_elapsed):
        heating_power = self.pid_controller.update(self.current_temperature, time_elapsed)
        specific_heat_capacity_water = 4186
        mass_water = 1
        temperature_change_due_to_heating = heating_power * time_elapsed / (mass_water * specific_heat_capacity_water)
        temperature_difference = self.current_temperature - self.ambient_temperature
        heat_loss_rate = self.heat_transfer_coefficient * self.surface_area * temperature_difference
        temperature_change_due_to_loss = -heat_loss_rate * time_elapsed / (mass_water * specific_heat_capacity_water)

        self.current_temperature += (temperature_change_due_to_heating + temperature_change_due_to_loss)

        if self.current_temperature > self.max_temperature:
            self.current_temperature = self.max_temperature

        print(heating_power)
        return self.current_temperature

pid = PIDController(kp=1, ki=0.01, kd=0.05, setpoint=60) 
heater = WaterHeaterWithPID(20,100,20,1,100,pid)
temp_list = []
time_list = np.linspace(0,1080,1000)
for _ in range(1000): 
    temp = heater.heat_water(80)
    temp_list.append(temp)
    print(f"Current temperature: {temp:.2f}°C")

plt.plot(time_list, temp_list)  
plt.xlabel('Time')  
plt.ylabel('Temp') 
plt.title('Temperature of Water vs Time')  
plt.show()  