import numpy
import pulp

N = 200 # how many timesteps
TIMESTEPS = range(N)
T_0 = 0.0
T_1 = 20.0

TIME = numpy.linspace(T_0, T_1, N)


print 'simulating true object trajectory'

# model the trajectory we want to track
# -- do some spiral in the plane, say
# -- pick some domain (square in R^2)

def object_pos(t):
    t_c = 0.5 * (T_0 + T_1)
    phase_one_x = 0.5 * t * numpy.cos(t)
    phase_one_y = 0.5 * t * numpy.sin(t)

    phase_two_x = 0.5 * t_c * numpy.cos(t_c) + 2.0 * numpy.abs(t - t_c) ** 1.1
    phase_two_y = 0.5 * t_c * numpy.sin(t_c)

    x = numpy.where(t < t_c, phase_one_x, phase_two_x)
    y = numpy.where(t < t_c, phase_one_y, phase_two_y)
    return (x, y)

object_x, object_y = object_pos(TIME)

print 'simulating sensor measurements'

# model sensor measurements
# -- dont start with any randomness
# -- place 8 sensors, say, randomly in the domain
# -- each sensor measures a reading of 1.0 / (||x-b||_1 + 1.0), say
# -- where x is the true object location and b is the sensor location

DOMAIN_SIZE = 25
DOMAIN_MIN = [-DOMAIN_SIZE, -DOMAIN_SIZE]
DOMAIN_MAX = [DOMAIN_SIZE, DOMAIN_SIZE]

N_SENSORS = 13
SENSORS = range(N_SENSORS)

sensor_x = numpy.random.uniform(DOMAIN_MIN[0], DOMAIN_MAX[0], N_SENSORS)
sensor_y = numpy.random.uniform(DOMAIN_MIN[0], DOMAIN_MAX[1], N_SENSORS)

SENSOR_POS = numpy.asarray([sensor_x, sensor_y])


def l1_sensor_observations(i):
    """i : sensor index"""
    dx = sensor_x[i] - object_x
    dy = sensor_y[i] - object_y
    err = numpy.abs(dx) + numpy.abs(dy)
    return (err + 1.0) ** -1.0


OBSERVATIONS = numpy.asarray([l1_sensor_observations(i) for i in SENSORS])


# set up an LP to try to recover the trajectory

# -- variables : need (x0, x1) object coords for each timestep
# -- for each timstep and detector index, need variables measuring x0 and x1 abs diff
#       --  between object location and detector location
#       --  penalise these by observation weights in the objective function
# -- for each pair of successive timesteps, need variables measuring x0 and x1 abs diff
#       --  between next and current object location
# -- need a variable to bound the max velocity per timestep
#       --  penalise large values of this in the objective function


print 'building LP for trajectory recovery'


prob = pulp.LpProblem('tracking', pulp.LpMaximize)

DIMS = range(2)

# variables::

# X[(t, d)] # d-th coord position at timestep t in [0, N)
X = {(t, d):pulp.LpVariable(('X', t, d), DOMAIN_MIN[d], DOMAIN_MAX[d]) for t in TIMESTEPS for d in DIMS}

# Z[(t, d, i)] # d-th coord negated absolute displacement from sensor location i at timestep t in [0, N)]
Z = {(t, d, i):pulp.LpVariable(('Z', t, d, i)) for t in TIMESTEPS for d in DIMS for i in SENSORS}

# V[(t, d)] # d-th coord negated absolute velocity at timestep t in [1, N)]
V = {(t, d):pulp.LpVariable(('V', t, d)) for t in TIMESTEPS[1:] for d in DIMS}

obj = []

for t in TIMESTEPS:
    for d in DIMS:
        for i in SENSORS:
            prob += (Z[(t, d, i)] <= X[(t, d)] - SENSOR_POS[d, i])
            prob += (Z[(t, d, i)] <= -(X[(t, d)] - SENSOR_POS[d, i]))
            obj.append(OBSERVATIONS[i, t] * Z[(t, d, i)])


MAX_VELOCITY = 3.0 # XXX
VELOCITY_PENALTY = 1.0 # bias toward slower trajectories
for t in TIMESTEPS[1:]:
    for d in DIMS:
        prob += (V[(t, d)] <= X[(t, d)] - X[(t-1, d)])
        prob += (V[(t, d)] <= -(X[(t, d)] - X[(t-1, d)]))
        prob += (V[(t, d)] >= -MAX_VELOCITY)
        obj.append(-VELOCITY_PENALTY * V[(t, d)])


prob += pulp.lpSum(obj)


print 'solving LP'
status = prob.solve(pulp.COIN(msg=0))

print 'finished solving:'

print pulp.LpStatus[status]
assert pulp.LpStatus[status] == 'Optimal'

if False:
    for t in TIMESTEPS:
        for d in DIMS:
            print 'X[%r] = %r' % ((t, d), pulp.value(X[(t, d)]))

if False:
    for t in TIMESTEPS:
        for d in DIMS:
            for i in SENSORS:
                print 'Z[%r] = %r' % ((t, d, i), pulp.value(Z[(t, d, i)]))

print 'recovering solution:'

inferred_object_x = numpy.asarray([pulp.value(X[(t, 0)]) for t in TIMESTEPS])
inferred_object_y = numpy.asarray([pulp.value(X[(t, 1)]) for t in TIMESTEPS])

print 'displaying plots'

import pylab
pylab.figure()

pylab.subplot(2, 2, 1)
pylab.title('true object trajectory + sensor locations')
pylab.scatter(object_x, object_y, s=96, c=TIME, cmap=pylab.cm.jet)
pylab.colorbar()
pylab.scatter(sensor_x, sensor_y, marker='+', s=48, c='m', linewidth=24)

pylab.subplot(2, 2, 2)
pylab.title('sensor observations')
pylab.plot(OBSERVATIONS.T)

pylab.subplot(2, 2, 3)
pylab.title('inferred object trajectory')
pylab.scatter(inferred_object_x, inferred_object_y, marker='p', s=96, c=TIME, cmap=pylab.cm.jet)
pylab.colorbar()

pylab.show()
