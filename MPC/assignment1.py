import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['FULL_RECALCULATE'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 20
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference = [50, 0, 0]  #[x_t , y_t, psi_t]

    def plant_model(self, prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        v_t = prev_state[3] # m/s
        # Define acceleration as the accel pedal position
        a_t = pedal
        # Equation for x_t & v_t at time t+1
        x_t_1 = x_t + (v_t * self.dt)
        v_t_1 = v_t + (a_t * self.dt) - (v_t / 25.0)

        return [x_t_1, 0, 0, v_t_1]

    def cost_function(self,u, *args):
        state = args[0] #state = [x_t, y_t, psi_t, v_t]
        ref = args[1] #ref = [x_t, y_t , psi_t]
        cost = 0.0
        # From class Q&A,u is a vector that has length equal to # of outputs multiplied by time horizon.
        # To write it out explicitly u would look like this where "_1" indicates time step 1):
        # u = [pedal_0, steering_0, pedal_1, steering_1, pedal_2, steering_2, .... pedal_n, steering_n].
        # So if you want the pedal from the 2nd timestep k = 2. You need to look at the u[4].
        # u[0] is pedal_0, u[1] is steering_0, u[2] is pedal_1, etc.

        for k in range(0, self.horizon):
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1])

            # Position cost
            cost += (ref[0] - state[0])**2

            # Speed Cost
            veh_speed = state[3] * 3.36 # Here v_t is in m/s. Need to convert to kmph
            if((veh_speed) > 10.0):
                # We start we a cost value of 50^2 = 2500. Hence we need to give a appropriate cost value
                # veh_speed * 100 will be above 1000. So good value to go with
                # Ideally needs to be smooth. There is a discontinuity here at 9.9 amd 10. This can be a good starting point
                cost += veh_speed * 10000

        return cost

sim_run(options, ModelPredictiveControl)
