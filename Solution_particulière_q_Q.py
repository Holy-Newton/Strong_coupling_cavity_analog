import numpy as np
import matplotlib.pyplot as plt
import math as m
from matplotlib.widgets import Slider

f = 1
wc, K = 1, 1 #Cavity frequency, "cavity diffusion"
D, T = 1.1, 1  #Emitter frequency, "diffusion"
Wl = 1 #laser frequency
Rabi = 1 #Rabi frequency
f = 1 # Force of the excitator
m = 1 # mass

time = 20 #s
t = np.linspace(0, time, time *10)

w_line = np.linspace(0, 5, 5*100)
w = 0.9

def A_phi_particulier(wc, Rabi, w, K, D, T, f, m): #retourne les amplitudes et phases de mes composantes de la solution particulière.
    gamma = -wc*Rabi
    a = wc**2 + wc*Rabi - w**2  #q
    b = w*K/2
    a_prime = D**2 + wc*Rabi -w**2  #Q
    b_prime = w*T/2
    alpha = a_prime*a - b_prime*b - gamma**2
    beta = a_prime*b + b_prime*a
    Aq = (f/m) * np.sqrt(a + b) / np.sqrt(alpha**2 + beta**2)
    AQ = (f/m) * (wc * Rabi)    / np.sqrt(alpha**2 + beta**2)

    phi_q = np.arctan((b*alpha-a*beta)/(a*alpha+b*beta))
    phi_Q = np.arctan(-beta/alpha)

    return Aq, AQ, phi_q, phi_Q

def q_Q_particulier(t, wc, Rabi, w, K, D, T, f, m):
    Aq, AQ, phi_q, phi_Q = A_phi_particulier(wc, Rabi, w, K, D, T, f, m)
    q_p = Aq * np.cos(w*t+phi_q)
    Q_p = AQ * np.cos(w*t+phi_Q)
    return q_p, Q_p


#sliderw = plt.axes([0.15, 0.25, 0.25, 0.03])  #slide valeurs w
#w_slide = Slider(sliderw, "w", 0.01, 3, valinit= 1)

plt.subplot(221)
plt.plot(t, q_Q_particulier(t, wc, Rabi, w, K, D, T, f, m)[0],"red")
plt.ylabel("q and Q particulier")
plt.xlabel("t")
plt.grid(True)

plt.subplot(221)
plt.plot(t, q_Q_particulier(t, wc, Rabi, w, K, D, T, f, m)[1],"blue")
plt.ylabel("particulier")
plt.xlabel("t")
plt.grid(True)

plt.subplot(222)
plt.plot(w_line, A_phi_particulier(wc, Rabi, w_line, K, D, T, f, m)[0],"red")
plt.ylabel("Amplitude q_p")
plt.xlabel("w")
plt.grid(True)

plt.subplot(224)
plt.plot(w_line, A_phi_particulier(wc, Rabi, w_line, K, D, T, f, m)[1],"blue")
plt.ylabel("Amplitude Q_p")
plt.xlabel("w")
plt.grid(True)

plt.subplot(223)
plt.plot(w_line, np.degrees(A_phi_particulier(wc, Rabi, w_line, K, D, T, f, m)[2]),"red")
plt.ylabel("phase")
plt.xlabel("w")
plt.grid(True)

plt.subplot(223)
plt.plot(w_line, np.degrees(A_phi_particulier(wc, Rabi, w_line, K, D, T, f, m)[3]),"blue")
plt.ylabel("phase")
plt.xlabel("w")
plt.grid(True)


plt.show()

