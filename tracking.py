import numpy
import pulp

N = 100 # how many timesteps
TIMESTEPS = range(N)
T_0 = 0.0
T_1 = 20.0

print 'using N=%r timesteps' % N

TIME = numpy.linspace(T_0, T_1, N)


print 'simulating true object trajectory'

# model the trajectory we want to track
# -- do some spiral in the plane, say
# -- pick some domain (square in R^2)

def object_pos(t):
    t_c = 0.5 * (T_0 + T_1)
    phase_one_x = 2.5 * t * numpy.cos(t)
    phase_one_y = 2.5 * t * numpy.sin(t)

    phase_two_x = 2.5 * t_c * numpy.cos(t_c) + 4.0 * numpy.abs(t - t_c) ** 1.1
    phase_two_y = 2.5 * t_c * numpy.sin(t_c) - 1.0 * numpy.abs(t - t_c) ** 1.5

    x = numpy.where(t < t_c, phase_one_x, phase_two_x)
    y = numpy.where(t < t_c, phase_one_y, phase_two_y)
    return (x, y)

object_x, object_y = object_pos(TIME)

print 'simulating sensor measurements'

# model sensor measurements
# -- each sensor measures a reading of 1.0 / (||x-b||_1 + 1.0), say
# -- where x is the true object location and b is the sensor location

DOMAIN_SIZE = 40
DOMAIN_MIN = [-DOMAIN_SIZE] * 2
DOMAIN_MAX = [DOMAIN_SIZE] * 2

N_SENSORS = 7 ** 2
SENSORS = range(N_SENSORS)

if True:
    # XXX uniform sensor grid appears to work much better than random.
    # think about this and L1 norm used for distances in LP
    print 'making uniform grid of %r sensors' % N_SENSORS
    SQRT_N_SENSORS = int(N_SENSORS ** 0.5)
    sensor_grid_x = numpy.linspace(DOMAIN_MIN[0], DOMAIN_MAX[0], SQRT_N_SENSORS)
    sensor_grid_y = numpy.linspace(DOMAIN_MIN[1], DOMAIN_MAX[1], SQRT_N_SENSORS)

    sensor_x, sensor_y = numpy.meshgrid(sensor_grid_x, sensor_grid_y)
    sensor_x = numpy.ravel(sensor_x)
    sensor_y = numpy.ravel(sensor_y)

else:
    print 'making %r randomly-placed sensors' % N_SENSORS
    sensor_x = numpy.random.uniform(DOMAIN_MIN[0], DOMAIN_MAX[0], N_SENSORS)
    sensor_y = numpy.random.uniform(DOMAIN_MIN[0], DOMAIN_MAX[1], N_SENSORS)

SENSOR_POS = numpy.asarray([sensor_x, sensor_y])


def sensor_observations(i):
    """i : sensor index"""
    dx = sensor_x[i] - object_x
    dy = sensor_y[i] - object_y
    distance = numpy.abs(dx) + numpy.abs(dy) # L1
    # distance = (dx**2 + dy**2) ** 0.5 # L2
    signal = (distance + 1.0) ** -1.0
    return signal

def noise_model(signal):
    noise = numpy.random.normal(0.0, 0.05)
    noisy_signal = numpy.maximum(noise + signal, 0.0)
    # maybe it doesnt even give measurements sometimes?
    pr_measurement = 0.50
    mask = numpy.random.uniform(0.0, 1.0, noisy_signal.shape) <= pr_measurement
    return numpy.where(mask, noisy_signal, 0.0)


NOISE_FREE_OBSERVATIONS = numpy.asarray([sensor_observations(i) for i in SENSORS])
OBSERVATIONS = numpy.asarray([noise_model(si) for si in NOISE_FREE_OBSERVATIONS])


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
print 'made %r variables for inferred object position' % len(X)

# Z[(t, d, i)] # d-th coord absolute distance from sensor location i at timestep t in [0, N)]
Z = {(t, d, i):pulp.LpVariable(('Z', t, d, i)) for t in TIMESTEPS for d in DIMS for i in SENSORS}
print 'made %r variables for object-sensor absolute distances' % len(Z)

# V[(t, d)] # d-th coord absolute velocity at timestep t in [1, N)]
V = {(t, d):pulp.LpVariable(('V', t, d)) for t in TIMESTEPS[1:] for d in DIMS}
print 'made %r variables for object velocities' % len(V)

obj = [] # objective function terms

for t in TIMESTEPS:
    for d in DIMS:
        for i in SENSORS:
            prob += (Z[(t, d, i)] >= X[(t, d)] - SENSOR_POS[d, i])
            prob += (Z[(t, d, i)] >= -(X[(t, d)] - SENSOR_POS[d, i]))
            #XXX this if condition is a hack to ignore non-close observations.
            # seems to make it work better. compare this cutoff against sensor observation plot
            # the cutoff value is important
            if OBSERVATIONS[i, t] >= 0.1:
                obj.append(-OBSERVATIONS[i, t] * Z[(t, d, i)])

MAX_VELOCITY = 5.0 # XXX
VELOCITY_PENALTY = 0.01 # bias a little toward slower trajectories?
for t in TIMESTEPS[1:]:
    for d in DIMS:
        prob += (V[(t, d)] >= X[(t, d)] - X[(t-1, d)])
        prob += (V[(t, d)] >= -(X[(t, d)] - X[(t-1, d)]))
        prob += (V[(t, d)] <= MAX_VELOCITY)
        obj.append(-VELOCITY_PENALTY * V[(t, d)])

prob += pulp.lpSum(obj)


print 'solving LP'
status = prob.solve(pulp.COIN(msg=1))
print 'solution status: %r' % pulp.LpStatus[status]
assert pulp.LpStatus[status] == 'Optimal'

print 'recovering solution'

inferred_object_x = numpy.asarray([pulp.value(X[(t, 0)]) for t in TIMESTEPS])
inferred_object_y = numpy.asarray([pulp.value(X[(t, 1)]) for t in TIMESTEPS])

print 'displaying plots'

import matplotlib
matplotlib.rc('font', family='Serif', weight='bold', size=12)
import pylab

pylab.figure(figsize=(14, 11))
pylab.suptitle('$L_1$ trajectory recovery')

pylab.subplot(2, 2, 1, axisbg='grey')
pylab.title('(a) True object trajectory + sensor locations.')
pylab.scatter(sensor_x, sensor_y, marker='+', s=18, c='m', linewidth=12)
pylab.scatter(object_x, object_y, s=96, c=TIME, cmap=pylab.cm.jet)
cbar = pylab.colorbar()
cbar.set_label('time $t$')
pylab.xlabel('position $x$')
pylab.ylabel('position $y$')

pylab.subplot(2, 2, 2)
pylab.title('(b) Noise-free sensor observations.')
sensor_scale = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=False) # make both observation plots use same scale.
pylab.imshow(NOISE_FREE_OBSERVATIONS, norm=sensor_scale, cmap=pylab.cm.hot, interpolation='none')
cbar = pylab.colorbar()
cbar.set_label('inverse $L_1$ distance')
pylab.xlabel('timestep index')
pylab.ylabel('sensor index')

pylab.subplot(2, 2, 4)
pylab.title('(c) Noisy partial sensor observations.')
pylab.imshow(OBSERVATIONS, norm=sensor_scale, cmap=pylab.cm.hot, interpolation='none')
cbar = pylab.colorbar()
cbar.set_label('inverse $L_1$ distance')
pylab.xlabel('timestep index')
pylab.ylabel('sensor index')

pylab.subplot(2, 2, 3, axisbg='grey')
pylab.title('(d) Inferred object trajectory + sensor locations.')
pylab.scatter(sensor_x, sensor_y, marker='+', s=18, c='m', linewidth=12)
pylab.scatter(inferred_object_x, inferred_object_y, marker='o', s=96, c=TIME, cmap=pylab.cm.jet)
cbar = pylab.colorbar()
cbar.set_label('time $t$')
pylab.xlabel('position $x$')
pylab.ylabel('position $y$')

if False:
    pylab.show()
else:
    pylab.savefig('out.png', bbox_inches='tight')

