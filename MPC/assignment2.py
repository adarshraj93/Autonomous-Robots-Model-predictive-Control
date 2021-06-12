import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = False

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 20
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 10, 0]
        self.reference2 = [10, 2, 3*3.14/2]

    def plant_model(self,prev_state, dt, pedal, steering):
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        beta = steering
        a_t = pedal

        # Updating the Motion Model
        x_t_1 = x_t + (v_t * np.cos(psi_t) * self.dt)
        y_t_1 = y_t + (v_t * np.sin(psi_t) * self.dt)
        v_t_1 = v_t + (a_t * self.dt) - (v_t / 25.0)
        psi_t_1 = psi_t + (v_t * (np.tan(beta)/2.50) * self.dt)

        return [x_t_1, y_t_1, psi_t_1, v_t_1]

    def cost_function(self,u, *args):
        state = args[0] #state = [x_t, y_t, psi_t, v_t]
        ref = args[1] #ref = [x_t, y_t , psi_t]
        cost = 0.0
        # From class Q&A,u is a vector that has length equal to # of outputs multiplied by time horizon.
        # To write it out explicitly u would look like this where "_1" indicates time step 1):
        # u = [pedal_0, steering_0, pedal_1, steering_1, pedal_2, steering_2, .... pedal_n, steering_n].
        # So if you want the pedal from the 2nd timestep k = 2. You need to look at the u[4].
        # u[0] is pedal_0, u[1] is steering_0, u[2] is pedal_1, etc.

        for k in range(0,self.horizon):
            v_start = state[3]
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1])

            # Position Cost: Here we need to take both x & y
            cost += abs(ref[0] - state[0])**2
            cost += abs(ref[1] - state[1])**2
            # Adding Linear cost, this takes a different path when going to position 2
            #cost += abs(ref[0] - state[0])
            #cost += abs(ref[1] - state[1])

            # Angle Cost:
            cost += abs(ref[2] - state[2])**2

            # Acceleration Cost: TO reduce the acceleration. There is tradeoff. Less accurate in the first position
            cost += (state[3] - v_start)**2*100

            #Steering Cost:
            #cost += u[k*2+1]**2*self.dt*100
        return cost

sim_run(options, ModelPredictiveControl)
