import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


q_0, q_p0 = 1, 0 # initial conditions q(0), q'(0)
wc = 1
w0 = 1.2*wc
 #Cavity frequency, Drvie frequency
K = wc/10 #"cavity diffusion" 

if w0 == 0:f = 0 # Force of the drive # (/ mass)   if w0 = 0 ==> f=0
else: f = 1 

time = 20*2*np.pi/wc #s représentation périodique

# Pour mes représentation graphiques 
w, t = np.linspace(0.5, 1.5, 500), np.linspace(0, time, 500)

#===Résolution Analytique===

# Im et Re de la solution particulière.
def Re_Im(wc, w, K, f):
    w0 = w
    gamma = ((wc**2-w0**2)**2+(K*w0/2)**2)
    re = f*(wc**2-w0**2)/gamma
    im = f*(-K*w0/2)/gamma   ### mais si utilisation dxse exp(-iwt): alors pic de résonnance im positif
    return re,im
# Angle phi de déphasage de la solution particulière
def phi(wc, w, K, f): 
    w0 = w
    re, im = Re_Im(wc, w0, K, f)
    Z = re + 1j*im
    return np.angle(Z)
#retourne l'amplitude de q_p(w)
def q_w_(wc, w, K): 
    w0 = w
    gamma = (wc**2-w0**2)**2 + (w0*K/2)**2
    q_w = f/np.sqrt(gamma)
    return q_w
# retourne q(t), l'oscillation temporelle:
def q(t, wc, w, K, f, q_0, q_p0):  
    w0= w
    phi_= phi(wc, w0, K, f)
    q_w = q_w_(wc, w0, K)
    q_particulier = q_w * np.cos(w0*t + phi_)

    q_p0_t0 = q_w * np.cos(phi_) ## t=0
    dq_p0_t0 = -w0 * q_w * np.sin(phi_)
    w_h= np.sqrt( wc**2 - (K**2)/16 )
    alpha = q_0 - q_p0_t0
    beta = (q_p0 - dq_p0_t0 + (K/4)*alpha) / w_h
    q_homogène = np.exp(-K/4*t)*(alpha * np.cos(w_h*t) + beta * np.sin(w_h*t))

    return q_particulier + q_homogène, q_particulier, alpha, beta


#===fonction pour résolution numérique avec solve_ivp===

def fun(t, S, w0, wc, K, f): 
    dq_dt = S[1]
    dp_dt = -K/2 * S[1] - wc**2 * S[0] + f * np.cos(w0 * t)
    return [dq_dt, dp_dt]
# Résolution ivp
solution = solve_ivp(fun, [0, time], [q_0, q_p0], method='RK45', max_step=0.1, args=(w0, wc, K, f))

#=========représentation grphique=========

# Fenêtre 1 : Analyse fréquentielle
fig, axs = plt.subplots(2,2,figsize=(11, 6.2), constrained_layout=True)

#Partie réelle
axs[0, 0].plot(w, Re_Im(wc, w, K, f)[0], color='blue')
axs[0, 0].axvline(wc, color='grey', linestyle='--', label='w = wc')
axs[0, 0].set_title("Real part of q_part(w)")
axs[0, 0].set_ylabel("Re[q(w)] Amplitude [A.U]")
axs[0, 0].set_xlabel("w [rad/t]")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Partie imaginaire
axs[1, 0].plot(w, Re_Im(wc, w, K, f)[1], color='red')
axs[1, 0].axvline(wc, color='grey', linestyle='--', label='w = wc')
axs[1, 0].set_title("Imaginary part of q_part(w)")
axs[1, 0].set_ylabel("Im[q(w)] Amplitude [A.U]")
axs[1, 0].set_xlabel("w[rad/t]")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Phase
axs[0, 1].plot(w, np.degrees(phi(wc, w, K, f)), color='black')
axs[0, 1].axvline(wc, color='grey', linestyle='--', label='w = wc')
axs[0, 1].set_title("Phase shift of q_part(w)")
axs[0, 1].set_ylabel("Phase shift in degrees")
axs[0, 1].set_xlabel("w [rad/t]")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Amplitude
axs[1, 1].plot(w, q_w_(wc, w, K), color='blue')
axs[1, 1].axvline(wc, color='grey', linestyle='--', label='w = wc')
axs[1, 1].set_title("Amplitude of q_part(w)")
axs[1, 1].set_ylabel("Amplitude [A.U]")
axs[1, 1].set_xlabel("w [rad/t]")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Fenêtre 2 : Oscillation temporelle
plt.figure(figsize=(7, 3))
plt.plot(solution.t*wc/(2*np.pi), solution.y[0], ":",linewidth=2.8, label="numeric RK45 q(T)", color="grey")
plt.plot(t*wc/(2*np.pi), q(t, wc, w0, K, f, q_0, q_p0)[0], linewidth=1.6, label="analytic q(T)", color="black")
#plt.plot(t*wc/(2*np.pi), q(t, wc, w0, K, f, q_0, q_p0)[1], "--",linewidth=1, label="q_particular(T)", color="grey")
#plt.title("q(T) periodic evolution")
plt.ylabel("q(T) [A.U]")
plt.xlabel("T [2π / wc]")
#plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()