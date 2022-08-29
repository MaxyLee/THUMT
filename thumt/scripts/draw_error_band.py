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
    'test2016': [36.76, 38.16, 38.24, 38.94],
    'test2017': [33.32, 33.48, 33.90, 34.51],
    'test2018': [31.65, 31.83, 32.10, 32.89],
    'testcoco': [27.78, 28.24, 29.16, 28.78],
}

vpt_stds = {
    'test2016': [0.14, 0.80, 0.63, 0.34],
    'test2017': [0.35, 0.36, 0.11, 0.15],
    'test2018': [0.45, 0.57, 0.41, 0.46],
    'testcoco': [0.61, 0.47, 0.62, 0.37],
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