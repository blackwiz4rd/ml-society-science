import matplotlib.pyplot as plt
import numpy

## We want to calculate teh average salary
n_people = 100 # The number of people participating
max_salary = 10 # maximum salary of people - otherwise we can't use Laplace
epsilon = 0.1 # the amount of privacy we want to lose

# Assume the distribution of people is coming from a clipped gamma distribution. Not necessarily true
data = numpy.random.gamma(shape=2, scale=1, size=n_people)
for i in range(n_people):
    data[i] = min(data[i], max_salary)

# Calculate the average salary
average_salary = numpy.average(data)
print("The actual average salary is ", average_salary)

#### Laplace mechanism for local DP
#
# We need the sensitivity of individual data points. Since an
# individual's data can vary at most by max_salary, we have:
def LocalDP(data, epsilon, max_salary, n_people):
    local_sensitivity = max_salary
    # We now tune the noise to the sensitivity
    local_noise = numpy.random.laplace(scale=local_sensitivity/epsilon, size=n_people)
    # Calculate the average
    local_average = numpy.average(data + local_noise)
    return local_average

#### Laplace mechanism for centralised DP
#
# We calculate the average, so if an individual's data changes by max_salary, the average
# changes by max_salary / n. So:
def CentralDP(data, epsilon, max_salary, n_people):
    central_sensitivity = max_salary / n_people
    # We now tune sensitivity to the function
    central_noise = numpy.random.laplace(scale=central_sensitivity / epsilon, size=1)
    # Calculate the average
    central_average = numpy.average(data + central_noise)
    return central_average

n_iter = 1000
variance_local = numpy.zeros(100)
variance_central = numpy.zeros(100)
epsilon = numpy.linspace(0.1,1,100) # 0.1 in steps of 100
for t in range(100):
    central = numpy.zeros(n_iter)
    local = numpy.zeros(n_iter)
    for i in range(n_iter):
        central[i] = CentralDP(data, epsilon[t], max_salary, n_people)
        local[i] = LocalDP(data, epsilon[t], max_salary, n_people)
    variance_local[t] = numpy.var(local)
    variance_central[t] = numpy.var(central)

plt.plot(epsilon, variance_local)
plt.plot(epsilon, variance_central)
plt.legend("Local", "Central")
plt.show()
