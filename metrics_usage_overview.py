#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:00:11 2019

@author: joshs
"""
# %%

plt.figure(417141)
plt.clf()

ok_units = (df['max_drift'] > 1)

plt.scatter(df[ok_units]['max_drift'],-df[ok_units]['DV_coordinate'],s=1, c='navy',alpha=0.15)
plt.ylim([-4.5,0])
plt.xlim([0,150])

# %%

plt.figure(417143)
plt.clf()

ok_units = (df['max_drift'] > 1)

plt.scatter(df[ok_units]['firing_rate'],-df[ok_units]['cortical_depth'],s=1, c='navy',alpha=0.15)
plt.ylim([-1,0])
plt.xlim([0,20])

# %%


plt.figure(417142)
plt.clf()

ok_units = (df['max_drift'] > 1) & \
           (df['isolation_distance'] < 1e3) & \
           (df['amplitude_cutoff'] < 0.5) & \
           (df['isolation_distance'] > 0)

plt.scatter(-np.log10(df[ok_units]['amplitude_cutoff']+0.000001),df[ok_units]['isolation_distance'],s=1, c='navy',alpha=0.10)
plt.ylim([-20,400])
#plt.xlim([0,150])

# %%

plt.figure(417144)
plt.clf()

ok_units = (df['silhouette_score'] > -1) & \
           (df['isolation_distance'] < 1e3) & \
           (df['amplitude_cutoff'] < 0.5) & \
           (df['isolation_distance'] > 0)

plt.scatter(-np.log10(df[ok_units]['l_ratio']+0.000001),df[ok_units]['silhouette_score'],s=1, c='navy',alpha=0.10)
plt.ylim([-0.2,0.5])

# %%

plt.figure(417145)
plt.clf()

ok_units = (df['silhouette_score'] > -1) & \
           (df['isolation_distance'] < 1e3) & \
           (df['amplitude_cutoff'] < 0.5) & \
           (df['nn_hit_rate'] > 0)

plot_density(df[ok_units]['d_prime'], df[ok_units]['nn_hit_rate'])

# %%

x = df[ok_units]['d_prime'].values
y = df[ok_units]['nn_hit_rate'].values

    # %%
    
from fastkde import fastKDE

N = pow(2,8) + 1

myPDF,axes = fastKDE.pdf(x,y, axes=(np.linspace(0,10,N), np.linspace(0,1,N)))

plt.clf()

#Extract the axes from the axis list
v1,v2 = axes
myPDF.shape

plt.imshow(myPDF,aspect='auto', origin='lower', vmax=3,cmap='Purples')

a = np.argmax(myPDF,1)
plt.plot(a,np.arange(N),color='slategrey')

#plt.xlim([0,15])
#plt.ylim([0,1])
# %%
    nbins=60
    k = kde.gaussian_kde([x,y])
    x_support = np.linspace(0,15,nbins)
    y_support = np.linspace(0,1,nbins)
    xx, yy = np.meshgrid(x_support, y_support)
    z = k([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.clf()
    plt.pcolormesh(xx, yy, z) #.reshape(xi.shape))

 
