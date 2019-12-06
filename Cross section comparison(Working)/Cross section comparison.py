#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import matplotlib.pyplot as plt


# In[79]:


def Rate_coefficient(Te):
    ##### Rate coefficient calculation #####
    # ref. R. K. Janev, et al., Elementary Processes in Hydrogen-Helium Plasmas, Springer (1987)
    
    # H + e -> H+ + 2e, Reaction 2.1.5 E = 13.6
    k1 = np.exp(-3.271396786375e+01+1.353655609057e+01*np.log(Te)-5.739328757388e+00*(np.log(Te))**2+1.563154982022e+00*(np.log(Te))**3-2.877056004391e-01*(np.log(Te))**4+3.482559773737e-02*(np.log(Te))**5-2.631976175590e-03*(np.log(Te))**6+1.119543953861e-04*(np.log(Te))**7-2.039149852002e-06*(np.log(Te))**8)
    # H+ + e -> H + hv, Reaction 2.1.8 E = Te
    k2 = 3.92e-14*(13.6/Te)**1.5/(13.6/Te+0.35) # n = 1s
    # H2 + e -> 2H + e, Reaction 2.2.5 E = 10
    k3 = np.exp(-2.858072836568e+01+1.038543976082e+01*np.log(Te)-5.383825026583e+00*(np.log(Te))**2+1.950636494405e+00*(np.log(Te))**3-5.393666392407e-01*(np.log(Te))**4+1.006916814453e-01*(np.log(Te))**5-1.160758573972e-02*(np.log(Te))**6+7.411623859122e-04*(np.log(Te))**7-2.001369618807e-05*(np.log(Te))**8)
    # H2 + e -> H2+ + 2e, Reaction 2.2.9 E = 15.4
    k4 = np.exp(-3.568640293666e+01+1.733468989961e+01*np.log(Te)-7.767469363538e+00*(np.log(Te))**2+2.211579405415e+00*(np.log(Te))**3-4.169840174384e-01*(np.log(Te))**4+5.088289820867e-02*(np.log(Te))**5-3.832737518325e-03*(np.log(Te))**6+1.612863120371e-04*(np.log(Te))**7-2.893391904431e-06*(np.log(Te))**8)
    # H2 + e -> H+ + H + 2e, Reaction 2.2.10 E = 18
    k5 = np.exp(-3.834597006782e+01+1.426322356722e+01*np.log(Te)-5.826468569506e+00*(np.log(Te))**2+1.727940947913e+00*(np.log(Te))**3-3.598120866343e-01*(np.log(Te))**4+4.822199350494e-02*(np.log(Te))**5-3.909402993006e-03*(np.log(Te))**6+1.738776657690e-04*(np.log(Te))**7-3.252844486351e-06*(np.log(Te))**8)
    # H2+ + e -> 2H+ + 2e, Reaction 2.2.11 E = 15.5
    k6 = np.exp(-3.746192301092e+01+1.559355031108e+01*np.log(Te)-6.693238367093e+00*(np.log(Te))**2+1.981700292134e+00*(np.log(Te))**3-4.044820889297e-01*(np.log(Te))**4+5.352391623039e-02*(np.log(Te))**5-4.317451841436e-03*(np.log(Te))**6+1.918499873454e-04*(np.log(Te))**7-3.591779705419e-06*(np.log(Te))**8)
    # H2+ + e -> H+ + H + e, Reaction 2.2.12 E = 10.5
    k7 = np.exp(-1.781416067709e+01+2.277799785711e+00*np.log(Te)-1.266868411626e+00*(np.log(Te))**2+4.296170447419e-01*(np.log(Te))**3-9.609908013189e-02*(np.log(Te))**4+1.387958040699e-02*(np.log(Te))**5-1.231349039470e-03*(np.log(Te))**6+6.042383126281e-05*(np.log(Te))**7-1.247521040900e-06*(np.log(Te))**8)
    # H2+ + e -> 2H, Reaction 2.2.14 E = Te
    k8 = np.exp(-1.670435653561e+01-6.035644995682e-01*np.log(Te)-1.942745783445e-08*(np.log(Te))**2-2.005952284492e-07*(np.log(Te))**3+2.962996104431e-08*(np.log(Te))**4+2.134293274971e-08*(np.log(Te))**5-6.353973401838e-09*(np.log(Te))**6+6.152557460831e-10*(np.log(Te))**7-2.025361858319e-11*(np.log(Te))**8)
    # H3+ + e -> H2 + H, Reaction 2.2.15 E = Te
    k9 = np.exp(-1.700270758355e+01-4.050073042947e-01*np.log(Te)+1.018733477232e-08*(np.log(Te))**2-1.695586285687e-08*(np.log(Te))**3+1.564311217508e-10*(np.log(Te))**4+1.979725412288e-09*(np.log(Te))**5-4.395545994733e-10*(np.log(Te))**6+3.584926377078e-11*(np.log(Te))**7-1.024189019465e-12*(np.log(Te))**8)
    # H3+ + e -> H+ + 2H + e, Reaction 2.2.16 E = 14
    k10 = np.exp(-3.078408636631e+01+1.509421488513e+01*np.log(Te)-7.349167207324e+00*(np.log(Te))**2+2.320966107642e+00*(np.log(Te))**3-4.818077551719e-01*(np.log(Te))**4+6.389229162737e-02*(np.log(Te))**5-5.161880953089e-03*(np.log(Te))**6+2.303985092606e-04*(np.log(Te))**7-4.344846146197e-06*(np.log(Te))**8)
    # H2+ + H2 -> H3+ + H, Reaction 4.3.3 E = 0
    k11 = 1.2194578959081685e-09  #for 0.3eV Hydrogen atom
    k12 = np.exp(-3.081902926338e+01+1.038866780735e+01*np.log(Te)-4.259768348687e+00*(np.log(Te))**2+1.181228673120e+00*(np.log(Te))**3-2.277513907465e-01*(np.log(Te))**4+2.900576728856e-02*(np.log(Te))**5-2.287591474628e-03*(np.log(Te))**6+1.004346442778e-04*(np.log(Te))**7-1.869930069131e-06*(np.log(Te))**8)
    return k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11*np.ones((len(Te))),k12


# In[80]:


NumReaction = 12
Te = np.arange(1e-1,1e3,1e-1)
name_list = []
for i in range(NumReaction):
    name_list.append('k'+ str(i+1))


# In[81]:


plt.figure(figsize=(20,20))
for i in range(NumReaction):
    plt.plot(Te,Rate_coefficient(Te)[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(b=True)
    plt.ylim(1e-12)
    plt.legend(name_list)
plt.show()


# In[ ]:




