# Setup


```python
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm
from tqdm import tqdm
warnings.filterwarnings("ignore")

import sklearn.linear_model as lm

os.chdir('../')
import dml_test
os.chdir('Example')
sns.set(font_scale=2,style='whitegrid')
np.random.seed(0)
```

## Loading STAR Dataset


```python
STAR_High_School = pd.read_spss('PROJECTSTAR/STAR_High_Schools.sav')
STAR_K3_School = pd.read_spss('PROJECTSTAR/STAR_K-3_Schools.sav').set_index('schid')
STAR_Students = pd.read_spss('PROJECTSTAR/STAR_Students.sav').set_index('stdntid')
Comparison_Students = pd.read_spss('PROJECTSTAR/Comparison_Students.sav').set_index('stdntid')

# pre-treatment covariates
gk_cols = list(filter(lambda x: 'gk' in x, STAR_Students.columns))
g1_cols = list(filter(lambda x: 'g1' in x, STAR_Students.columns))
g2_cols = list(filter(lambda x: 'g2' in x, STAR_Students.columns))
g3_cols = list(filter(lambda x: 'g3' in x, STAR_Students.columns))
g_cols = gk_cols+g1_cols+g2_cols+g3_cols

personal_cols = ['gender','race','birthyear']

cols_cond = ['surban',
            'tgen',
            'trace',
            'thighdegree',
            'tcareer',
            'tyears',
            'classsize',
            'freelunch']

class_sizes = ['g1classsize',
             'g2classsize']

g3scores = ['g3treadss',
            'g3tmathss',
            'g3tlangss',
            'g3socialsciss']

g_cols_cond = list(filter(lambda s: np.sum(list(map(lambda x: x in s,cols_cond)))>0,g_cols))
df_exp = STAR_Students[personal_cols]#+class_sizes]
df_exp['Sample'] = 1
df_exp['g3avgscore'] = STAR_Students[g3scores].mean(axis=1)
df_exp['g3smallclass'] = (STAR_Students['g3classsize']<=17).astype(int)

df_obs = Comparison_Students[personal_cols]#+class_sizes]
df_obs['Sample'] = 0
df_obs['g3avgscore'] = Comparison_Students[g3scores].mean(axis=1)
df_obs['g3smallclass'] = (Comparison_Students['g3classsize']<=17).astype(int)

df = df_exp.append(df_obs)
df_no_na = df.dropna()

df_no_na_dummified = pd.get_dummies(df_no_na,columns=['gender','race'],drop_first=True)
```

# Test


```python
N = df_no_na_dummified.shape[0]
df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df = dml_test.fit(Y='g3avgscore',T='g3smallclass', S='Sample',
                                                            df=df_no_na_dummified,
                                                            n_splits = 5)
psi = dml_test.Psi(df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df, 'g3avgscore', 'g3smallclass','Sample')
p_val = dml_test.dml_pval( psi)
```


```python
print('p(1) = %.4f, p(0) = %.4f'%(p_val[0],p_val[1]))
```

    p(1) = 0.0005, p(0) = 0.2408


# ATE Estimation


```python
te = dml_test.Lambda(df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df, 'g3avgscore','g3smallclass','Sample')
```


```python
print(r"ATE: %.4f ± %.4f"%(np.mean(te,axis=0)[0]- np.mean(te,axis=0)[1], 1.96*(np.std(te,axis=0)[0] + np.std(te,axis=0)[1])/np.sqrt(df.shape[0])))
```

    ATE: 5.7493 ± 2.6229

