import jax.numpy as jnp
from jax import config
#config.update("jax_platform_name", "gpu")
config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
#from microjax.fastlens.fftlog_org import *
from microjax.fastlens.fftlog_jax import fftlog as fftlog_jax
from microjax.fastlens.fftlog_jax import hankel as hankel_jax
from microjax.fastlens.fftlog import fftlog,hankel


print('This is a test of fftlog module written by Xiao Fang.')
print('nu is required to be between -ell to 2.')
k, pk = np.loadtxt('tests/Pk_test', usecols=(0,1), unpack=True)
N = k.size
print('number of input data points: '+str(N))
ell = 100
nu = 1.01
myfftlog = fftlog(k, pk, nu=nu, N_extrap_low=1500, N_extrap_high=1500, c_window_width=0.25, N_pad=5000)
r, Fr = myfftlog.fftlog(ell)
myfftlog_jax = fftlog_jax(k, pk, nu=nu, N_extrap_low=1500, N_extrap_high=1500, c_window_width=0.25, N_pad=5000)
r_j, Fr_j = myfftlog_jax.fftlog(ell)

################# Test fftlog ##############
print('Testing fftlog')
fig = plt.figure(figsize=(8,4))
fig.suptitle(r'$F(y) = \int_0^\infty f(x)j_{\ell}(xy) dx/x, \ell=$%.1f'%(ell))

subfig1 = fig.add_subplot(1,2,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
plt.tight_layout()

subfig2 = fig.add_subplot(1,2,2)
subfig2.set_title(r'$\nu=$%.2f'%(nu))
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel('F(y)')
subfig2.plot(r, Fr, label='fftlog')
subfig2.plot(r_j, Fr_j, label='fftlog_jax',ls="--")

# r_c, Fr_c = np.loadtxt('../cfftlog/test_output.txt', usecols=(0,1), unpack=True)
# subfig2.plot(r_c, Fr_c, label='(bad) brute-force')

# r_bf, Fr_bf = np.loadtxt('test_bruteforce.txt', usecols=(0,1), unpack=True)
# subfig2.plot(r_bf, Fr_bf)
plt.legend()
plt.tight_layout()
plt.show()

################# Test j' ##############
print('Testing 1st & 2nd-derivative')

r1, Fr1 = myfftlog.fftlog_dj(ell)
r2, Fr2 = myfftlog.fftlog_ddj(ell)
r1_j, Fr1_j = myfftlog_jax.fftlog_dj(ell)
r2_j, Fr2_j = myfftlog_jax.fftlog_ddj(ell)
fig = plt.figure(figsize=(8,4))
fig.suptitle(r'$F(y) = \int_0^\infty f(x)j_{\ell}^{(n)}(xy) dx/x, \ell=$%.1f, n=1,2'%(ell))

subfig1 = fig.add_subplot(1,2,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
plt.tight_layout()

subfig2 = fig.add_subplot(1,2,2)
subfig2.set_title(r'$\nu=$%.2f'%(nu))
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel('F(y)')
subfig2.plot(r1, abs(Fr1), label="1st-derivative",c="k")
subfig2.plot(r2, abs(Fr2), '-', label='2nd-derivative',c="blue")
subfig2.plot(r1_j, abs(Fr1_j), ":", label="1st-derivative jax",c="orange")
subfig2.plot(r2_j, abs(Fr2_j), ':', label='2nd-derivative jax',c="cyan")
# r_bf, Fr_bf = np.loadtxt('test_bruteforce.txt', usecols=(0,1), unpack=True)
# subfig2.plot(r_bf, Fr_bf)
plt.legend()
plt.tight_layout()
plt.show()

################# Test j squared ##############
print('Testing squared j')

r1, Fr1 = myfftlog.fftlog_jsqr(ell)
r1_j, Fr1_j = myfftlog_jax.fftlog_jsqr(ell)
fig = plt.figure(figsize=(8,4))
fig.suptitle(r'$F(y) = \int_0^\infty f(x)|j_{\ell}(xy)|^2 dx/x, \ell=$%.1f'%(ell))

subfig1 = fig.add_subplot(1,2,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
plt.tight_layout()

subfig2 = fig.add_subplot(1,2,2)
subfig2.set_title(r'$\nu=$%.2f'%(nu))
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel('F(y)')
subfig2.plot(r1, abs(Fr1))
subfig2.plot(r1_j, abs(Fr1_j),"--",label="jax")
plt.legend()
# r_bf, Fr_bf = np.loadtxt('test_bruteforce.txt', usecols=(0,1), unpack=True)
# subfig2.plot(r_bf, Fr_bf)
# plt.legend()
plt.tight_layout()
plt.show()

################# Test Hankel ##############
print('Testing hankel')

n = 0
nu = 1.01
myhankel = hankel(k, pk, nu=nu, N_extrap_low=1500, N_extrap_high=1500, c_window_width=0.25)
myhankel_jax = hankel_jax(k, pk, nu=nu, N_extrap_low=1500, N_extrap_high=1500, c_window_width=0.25)
r, Fr = myhankel.hankel(n)
r_j, Fr_j = myhankel_jax.hankel(n)

fig = plt.figure(figsize=(8,4))
fig.suptitle(r'$F(y) = \int_0^\infty f(x)J_{n}(xy) dx/x, n=$%d'%(n))

subfig1 = fig.add_subplot(1,2,1)
subfig1.set_xscale('log')
subfig1.set_yscale('log')
subfig1.set_xlabel('x')
subfig1.set_ylabel('f(x)')
subfig1.plot(k, pk)
plt.tight_layout()

subfig2 = fig.add_subplot(1,2,2)
subfig2.set_title(r'$\nu=$%.2f'%(nu))
subfig2.set_xscale('log')
subfig2.set_yscale('log')
subfig2.set_xlabel('y')
subfig2.set_ylabel('F(y)')
subfig2.plot(r, Fr)
subfig2.plot(r_j, Fr_j,"--",label="jax")
plt.legend()
plt.tight_layout()
plt.show()


#from microjax.fastlens import mag_fft

