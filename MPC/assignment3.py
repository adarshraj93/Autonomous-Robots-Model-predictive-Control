import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = True

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 20
        self.dt = 0.2

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 0, 0]
        self.reference2 = None

        self.x_obs = 5
        self.y_obs = 0.1

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
        psi_t_1 = psi_t + (v_t * (np.tan(beta) / 2.50) * self.dt)

        return [x_t_1, y_t_1, psi_t_1, v_t_1]


    def cost_function(self,u, *args):
        state = args[0] #state = [x_t, y_t, psi_t, v_t]
        ref = args[1] #ref = [x_t, y_t , psi_t]
        cost = 0.0

        for k in range(0, self.horizon):
            state = self.plant_model(state, self.dt, u[k*2], u[k*2+1])

            #Position cost
            cost += abs(ref[0] - state[0])**2
            cost += abs(ref[1] - state[1])**2

            #Steering cost
            cost += abs(ref[2] - state[2])**2

            #Obstacle Cost
            cost += self.obstacle_cost(state[0], state[1]) #Pass the x & y coordinates of the car
        return cost

    def obstacle_cost(self, x, y):
        distance = np.sqrt((x - self.x_obs)**2 + (y - self.y_obs)**2) #Distance of car from obstacle position
        if (distance > 2):
            return 15
        else:
            obs_cost = (1/distance)*30
            return obs_cost

sim_run(options, ModelPredictiveControl)
