import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

testname = 'testcoco'

ft_means = {
    'test2016': [36.22, 37.61, 38.05, 39.04],
    'test2017': [33.06, 33.46, 34.40, 35.10],
    'test2018': [31.37, 31.89, 32.28, 33.04],
    'testcoco': [27.68, 28.93, 29.66, 29.70],
}

ft_stds = {
    'test2016': [0.90, 0.41, 1.05, 0.18],
    'test2017': [0.16, 0.24, 0.28, 0.55],
    'test2018': [0.37, 0.33, 0.45, 0.14],
    'testcoco': [0.72, 0.43, 0.55, 0.83],
}

vpt_means = {
    'test2016': [36.45, 37.80, 38.03, 38.88],
    'test2017': [33.15, 33.47, 34.04, 34.13],
    'test2018': [31.62, 31.84, 32.21, 32.77],
    'testcoco': [28.05, 28.15, 28.65, 28.97],
}

vpt_stds = {
    'test2016': [0.41, 0.96, 0.63, 0.53],
    'test2017': [0.34, 0.30, 0.16, 0.48],
    'test2018': [0.28, 0.56, 0.41, 0.32],
    'testcoco': [0.60, 0.37, 0.51, 0.42],
}

sns.set()

x = np.array([50, 100, 200, 500])

mean_1 = np.array(ft_means[testname])
std_1 = np.array(ft_stds[testname])

mean_2 = np.array(vpt_means[testname])
std_2 = np.array(vpt_stds[testname])

plt.plot(x, mean_1, 'C0-', marker='o', label='FT')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='C0', alpha=0.2)
plt.plot(x, mean_2, 'C1-', marker='o', label='VPT')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='C1', alpha=0.2)

plt.xlabel('training data size')
plt.ylabel('BLEU-4')

plt.legend(title='method', loc='lower right')
plt.savefig(f'{testname}.png', dpi=300)