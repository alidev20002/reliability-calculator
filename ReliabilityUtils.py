import math
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, root
from scipy.stats import poisson

def estimate_goel_okumoto(data, t):
    X = np.array([[item["failure_rate"]] for item in data])
    y = np.array([item["cumulative_failures"] for item in data])

    model = LinearRegression()
    model.fit(X, y)

    a = model.intercept_
    b = abs(1.0 / model.coef_[0])
    total_time = data[-1]['cumulative_time']

    f = a * b * math.exp(-b * total_time)

    return math.exp(-f * t)

def estimate_weibull(data, t):
    # ti: time intervals, ki: number of faults observed
    # time intervals t_i
    ti = np.array([item["cumulative_time"] for item in data]) 
    # corresponding fault counts k_i  
    ki = np.array([item["failures"] for item in data])   

    n = len(ti)
    ti_1 = np.roll(ti, 1)
    ti_1[0] = 0  # t_0 = 0

    def equations(vars):
        b, c = vars
        term1 = 0
        term2 = 0
        sum_ki = np.sum(ki)

        for i in range(n):
            num1 = ki[i] * ((ti[i] ** c) * np.exp(-b * ti[i] ** c) - (ti_1[i] ** c) * np.exp(-b * ti_1[i] ** c))
            den1 = np.exp(-b * ti_1[i] ** c) - np.exp(-b * ti[i] ** c)
            term1 += num1 / den1

            num2 = ki[i] * (
                (ti[i] ** c) * np.log(ti[i]) * np.exp(-b * ti[i] ** c)
                - (ti_1[i] ** c) * np.log(ti_1[i] + 1e-10) * np.exp(-b * ti_1[i] ** c)
            )
            den2 = np.exp(-b * ti_1[i] ** c) - np.exp(-b * ti[i] ** c)
            term2 += num2 / den2

        eq1 = term1 - (ti[n-1] ** c) * np.exp(-b * ti[n-1] ** c) * sum_ki / (1 - np.exp(-b * ti[n-1] ** c))
        eq2 = term2 - b * (ti[n-1] ** c) * np.log(ti[n-1]) * np.exp(-b * ti[n-1] ** c) * sum_ki / (1 - np.exp(-b * ti[n-1] ** c))

        return [eq1, eq2]

    # Initial guesses for b and c
    initial_guess = [0.01, 1.0]
    sol = root(equations, initial_guess)

    if sol.success:
        b, c = sol.x
        denom = 1 - np.exp(-b * ti[-1] ** c)
        a = np.sum(ki) / denom
        print(f"تخمین پارامترها:\na = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
    else:
        print("حل معادلات همگرا نشد.")
        return 0, 'در حال حاضر استفاده از این مدل توصیه نمی‌شود'
    
    total_time = data[-1]['cumulative_time']
    
    f = (a * b * c) * ((b * total_time) ** (c-1)) * math.exp(-((b * total_time) ** c))
    return math.exp(-f * t), None

def estimate_log_logistics(data, t):
    
    def F_model(t, a, b, c):
        return (a * (b * t)**c) / (1 + (b * t)**c)
    
    initial_guess = [35, 0.1, 2]
    cumulative_time = np.array([item["cumulative_time"] for item in data])
    cumulative_failures = np.array([item["cumulative_failures"] for item in data])
    params, _ = curve_fit(F_model, cumulative_time, cumulative_failures, p0=initial_guess, bounds=(0, np.inf))

    a, b, c = params
    total_time = data[-1]['cumulative_time']
    f = (a * b * c * ((b * total_time) ** (c - 1))) / ((1 + ((b * total_time) ** c)) ** 2)

    return math.exp(-f * t)

def estimate_duane(data, t):
    cumulative_time = np.array([item["cumulative_time"] for item in data])
    cumulative_failures = np.array([item["cumulative_failures"] for item in data])

    ln_t = np.log(cumulative_time).reshape(-1, 1)
    ln_f = np.log(cumulative_failures)

    model = LinearRegression()
    model.fit(ln_t, ln_f)

    b = model.coef_[0]
    ln_a = model.intercept_
    a = np.exp(ln_a)
    total_time = data[-1]['cumulative_time']
    f = a * b * (total_time ** (b - 1))
    
    return math.exp(-f * t)

def test_and_estimation_reliability(total_failures, total_time, t):
    failure_rate = float(total_failures) / total_time
    return math.exp(-failure_rate * t)