import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.signal import find_peaks
import math
h_bar = 1.05457e-34
e_v = 1.602e-19
e_mass = 9.1093837e-31

Ewc = 0.0591 # eV
wc = 1 # Ewc/h_bar*e_v #3.34e15 #Cavity frequency
El = 1
El_coef, Rabi = 1 , 0.22*wc # Amplitude of electric field El/e_mass ?, #Rabi frequency

Delta = wc*0.00
D = wc+Delta  #Emitter frequency, "diffusion"
Q = 10 #quality[10;100]
K = 1/Q  #"cavity diffusion"
Γ = K * 5e-4 # (1/25e-12)/(8.038e13) #K/2 #(D-wc)*25e-12 #Delta/(wc/25e-12)
print("=================gamma is", Γ, K)
w = np.linspace(0.9*wc, 1.3*wc,1000)#laser frequency 
Delta_line = np.linspace(0, 8*Delta, 300)



#==========PARAMETERS OF THE PARΓICULAR SOLUΓION==========
def solve_deg_4(wc, D, K, Γ, Rabi):
    a = (-wc**2 - wc*Rabi)
    b = (-D**2 - wc*Rabi)
    a4 = 1
    a3 = 1j*(Γ+K)/2
    a2 = - b - a - Γ*K/4
    a1 = -1j*K/2*b - 1j*Γ/2*a
    a0 = a*b

    # Coefficients doivent être SCALAIRES
    coeffs = [a4, a3, a2, a1, a0]
    racines = np.roots(coeffs)
    racines_sorted = sorted(racines)  # pour l’ordre
    return_list = []
    for z in racines_sorted:
        if z.imag >= 0:
            return_list.append(z.imag)
    return return_list

def eigen_space(wc, D, K, Γ, Rabi):
    wc = np.sqrt(D**2+(K/4)**2-(Γ/4)**2)
    matrice = np.array([[0, -1, 0, 0],
                       [wc**2+wc*Rabi, +K/2, -wc*Rabi, 0],
                       [0, 0, 0, -1],
                       [-wc*Rabi, 0, D**2+wc*Rabi, Γ/2]])
    eigenvalues, eigenvectors = np.linalg.eig(matrice)
    return eigenvalues, eigenvectors

# return re and im parts for each oscillator
def Re_Im(El_coef, wc, D, K, Γ, Rabi, wl ):
    a = wc**2 + Rabi*wc - wl**2
    b = wl*K/2
    a_p = D**2 + Rabi*wc - wl**2
    b_p = wl*Γ/2
    gamma = -wc*Rabi
    alpha = a*a_p - b*b_p - gamma**2
    beta = b_p*a + b*a_p

    req = El_coef*( a_p*alpha + b_p* beta)/ (alpha **2 + beta**2)
    imq = El_coef*( b_p*alpha - a_p* beta)/ (alpha **2 + beta**2)

    reQ = El_coef*(-gamma * alpha)/ (alpha **2 + beta**2)
    imQ = El_coef*(gamma * beta)/ (alpha **2 + beta**2)

    return req, imq, reQ, imQ

#return the amplitudes of the drive particular solution 
def q_Q_w(El_coef, wc, D, K, Γ, Rabi, wl ):

    req, imq, reQ, imQ = Re_Im(El_coef, wc, D, K, Γ, Rabi, wl )
    return np.sqrt(req**2 + imq**2), np.sqrt(reQ**2 + imQ**2)

#return the phase of each oscillator
def dephasage(El_coef, wc, D, K, Γ, Rabi, wl ):

    req, imq, reQ, imQ = Re_Im(El_coef, wc, D, K, Γ, Rabi, wl )
    
    return np.angle(req + 1j * imq), np.angle(reQ + 1j * imQ)


#==================== numeric resolution ========================
def fun_ivp(t, z, El_coef, wc, D, K, Γ, Rabi, wl):
    q, dq_dt, Q, dQ_dt = z
    q_drive = El_coef* np.cos(wl * t)
    Q_drive = 0
    dq_2dt = -(wc**2 +wc*Rabi)*q -(K/2) * dq_dt + wc*Rabi* Q + q_drive
    dQ_2dt = -(D**2 +wc*Rabi)*Q -(Γ/2) * dQ_dt + wc*Rabi * q + Q_drive
    return [dq_dt, dq_2dt, dQ_dt, dQ_2dt]

def rabi_splitting(El_coef, wc, D, K, Γ, Rabi, wl):
    intensities = []
    for i in wl:
        re, im, reQ, imQ = Re_Im(El_coef, wc, D, K, Γ, Rabi, i)
        val = re**2 + im**2
        intensities.append(val)
    # Détection des pics
    peaks, a = find_peaks(intensities)
    if len(peaks)< 2:
        print("No rabi splitting")
        return

    # On garde les deux plus gros
    sorted_peaks = sorted(peaks, key=lambda x: intensities[x], reverse=True)
    w1 = wl[sorted_peaks[0]]
    w2 = wl[sorted_peaks[1]]
    splitting = abs(w2 - w1)
    max_val = max(intensities)
    print(f"===============Rabi splitting value:{splitting} ============")
    return splitting, w1, w2, max_val

#=============== single oscillator==============

# Im et Re de la solution particulière.
def Re_im_q(wc, w, K, El_coef):
    w0 = w
    gamma = ((wc**2-w0**2)**2+(K*w0/2)**2)
    re = El_coef*(wc**2-w0**2)/gamma
    im = El_coef*(K*w0/2)/gamma   ### mais si utilisation dxse exp(-iwt): alors pic de résonnance im positif
    return re,im

def analytic_simplification_re_im(wc, w, K, El_coef):
    gamma = ((wc-w)**2+(K/4)**2)
    re = El_coef/(2*wc)*(wc-w)/gamma
    im = El_coef/(2*wc)*(K/4)/gamma
    return re,im

def analytic_coupled_intensity(El_coef, wc, D, K, Γ, Rabi):
    im1 = Rabi/2 + (D - wc + (Γ-K)/4)**2/(4*Rabi)
    im2 = - Rabi/2 + (D - wc + (Γ-K)/4)**2/(4*Rabi)
    re = -1/2*(wc + D + 2*Rabi +(K+Γ)/4)
    return im1, im2, re


def analytic_simplification_q_w( El_coef, w, wc, D, K, Γ, Rabi):
    wc = np.sqrt(wc**2-(K/4)**2)
    Z_up = El_coef * (w+D+1j*Γ/4+Rabi)
    Z_dw = 2*wc*((wc+w+1j*K/4+Rabi)*(D+w+1j*Γ/4+Rabi)-Rabi**2)
    Z_q = Z_up/Z_dw
    return abs(Z_q)

def strong_coupling_w(wc, K, Γ, Rabi):
    wc = np.sqrt(wc**2-(K/4)**2)
    w_plus = np.sqrt(Rabi**2-((K-Γ)/4)**2)/2 + wc + Rabi/2
    w_minus = - np.sqrt(Rabi**2-((K-Γ)/4)**2)/2 + wc + Rabi/2
    lifetime = ((K+Γ)/8)
    return w_plus, w_minus, lifetime

def weak_coupling_w(wc, K, Γ, Rabi):
    w = wc + Rabi/2
    lifetime_plus = ((K+Γ)/8) + np.sqrt(-Rabi**2+((K-Γ)/4)**2)/2
    lifetime_minus = ((K+Γ)/8) - np.sqrt(-Rabi**2+((K-Γ)/4)**2)/2
    return w, lifetime_minus, lifetime_plus
'''
def eigen_frequency_grid(wc, D, K, Γ, N, x_max):
    Max_w = strong_coupling_w(wc, K, Γ, x_max*wc)[0]
    Min_w = weak_coupling_w(wc, K, Γ, 0)[0]
    rabi_vals = np.linspace(0, x_max*wc, N)
    w_vals = np.linspace(Min_w, Max_w, N)
    U = np.zeros((N, N))  # matrix

    for i, rabi_val in enumerate(rabi_vals):
        for k in range(200):
            w1, w2= solve_deg_4(wc, D, K, Γ, rabi_val)
            freqs = [w1, w2]
            for w in freqs:
                j = np.searchsorted(w_vals, w) - 1
                if 0 <= j < N:
                    U[i, j] += 1
    return U, rabi_vals, w_vals

'''
def analytic_resolution_w_rabi(Rabi_line, Rabi_line_sc, wc, K, Γ): 
    list_w1 = []
    list_w2 = []
    for rabi in Rabi_line:
        list_pos = []
        eigenvalues, eigenvectors = eigen_space(wc, wc, K, Γ, rabi)
        eigenvalues = sorted(eigenvalues)
        for z in eigenvalues:
            if z.imag >= 0:
                list_pos.append(z.imag)
        list_pos = sorted(list_pos)
        list_w1.append(list_pos[0]/wc)
        list_w2.append(list_pos[1]/wc)

    list_w1_sc = []
    list_w2_sc = []
    for rabi in Rabi_line_sc:
        list_pos = []
        eigenvalues, eigenvectors = eigen_space(wc, wc, K, Γ, rabi)
        for z in eigenvalues:
            if z.imag >= 0:
                list_pos.append(z.imag)
        list_pos = sorted(list_pos)
        list_w1_sc.append(list_pos[0]/wc)
        list_w2_sc.append(list_pos[1]/wc)
    
    list_w1_G = []
    list_w2_G = []
    for rabi in Rabi_line:
        list_pos_G = []
        eigenvalues, eigenvectors = eigen_space(wc, wc, K, Γ, rabi)
        for z in eigenvalues:
            list_pos_G.append(z.real)
        sort_list_G = sorted(list_pos_G)
        list_w1_G.append(sort_list_G[0]/wc)
        list_w2_G.append(sort_list_G[2]/wc)

    list_w1_sc_G = []
    list_w2_sc_G = []
    for rabi in Rabi_line_sc:
        list_pos_G = []
        eigenvalues, eigenvectors = eigen_space(wc, wc, K, Γ, rabi)
        for z in eigenvalues:
            list_pos_G.append(z.real)
        sort_list_G = sorted(list_pos_G)
        list_w1_sc_G.append(sort_list_G[0]/wc)
        list_w2_sc_G.append(sort_list_G[2]/wc)

    return list_w1, list_w2, list_w1_sc, list_w2_sc, list_w1_G, list_w2_G, list_w1_sc_G, list_w2_sc_G











#===============================================

fig, axs = plt.subplots(2,1,figsize=(4.7, 7.7), constrained_layout=True)
'''
# Phase
axs[0].plot(w/wc, np.unwrap(dephasage(El_coef, wc, D, K, Γ, Rabi, w )[0]),linewidth = "2", color='black', label = r"$\phi$[q($\omega_{L}$)]")
axs[0].plot(w/wc, np.unwrap(dephasage(El_coef, wc, D, K, Γ, Rabi, w )[1]),linewidth = "2", color='grey', label = r"$\phi$[Q($\omega_{L}$]")
#axs[0].set_title("Phase shifts in particular parts of the pendulum system")
axs[0].set_ylabel("Phase shift [radian]")
axs[0].set_xlabel("$\omega_{L}$ [$\omega_{C}$]")
axs[0].text(1.26, -1.5 , '(a)', fontsize=15.5, color = "black")
axs[0].grid(False)
axs[0].legend()
'''
# Intensité
splitting, w1, w2, max_val = rabi_splitting(El_coef, wc, D, K, Γ, Rabi, w ) 
splitting_visual = np.linspace(w1, w2, 40)
splitting_points = []
line_1 = []
print(w1, w2 ,"==========w1, w2============")
for i in splitting_visual:
    splitting_points.append(10)
    line_1.append(1/wc)
#u = 1.01
#for i in range(2):
#    axs[0].plot(w/wc, q_Q_w(El_coef, wc, u*wc, K, Γ, Rabi, w )[0]**2, "r--",color='grey')
#    u += 0.09
#u += 0.1*Rabi
#axs[1].plot(w/wc, q_Q_w(El_coef, wc, D, K, Γ, u, w )[0]**2, "r--",color='grey')
    
axs[1].plot(splitting_visual/wc, splitting_points, ":" ,color='red', label = r"$\Omega_{R}$"f"= {splitting/wc:.3f}[wc]",linewidth=2.5) #splitting*h_bar/e_v

#axs[1].plot(w/wc, q_Q_w(El_coef, wc, D, K, Γ, Rabi, w )[0]**2, color='grey', label = "I[q(w)]")
axs[1].plot(w/wc, q_Q_w(El_coef, wc, D, K, Γ, Rabi, w )[0]/El, color='black',linewidth = "1.5", label=r"|$\chi$($\omega_{L}$)|")#, label = r"$I_{q}$($\omega_{L}$) $[\frac{E_{L}(\omega_{L})}{E_{0}}]^2$")
#axs[1].plot(w/wc, (Re_im_q(D, w, Γ, El_coef)[0]**2+Re_im_q(D, w, Γ, El_coef)[1]**2),"r--" ,color='red', label = "Emitter intensity")

#axs[1].set_title("Intensity of the coupled cavity E-M field drived with an $\omega$laser frequency")
axs[1].set_ylabel(r"|$\chi$($\omega_{L}$)| $\propto$ Transmission")
axs[1].set_xlabel(r"$\omega_{L}$ [$\omega_{C}$]")
axs[1].text(0.94, 15 , 'P-', fontsize=18, color = "black")
axs[1].text(1.24, 15 , 'P+', fontsize=18, color = "black")
axs[1].text(0.92, 18 , '(b)', fontsize=12.5, color = "black", fontweight='bold')
axs[1].text(1.08, 11 , r'$\Omega_{R}$', fontsize=15, color = "red", fontweight='bold')
axs[1].set_xlim(0.9, 1.3)
axs[1].legend()
axs[1].grid(False)



fact_delta_max, fact_W_max = 0.5, 1.5
N = 300
w_1 = np.linspace(0.6*D, fact_W_max*D, N)
Delta_fig_1 = np.linspace(-fact_delta_max*D, fact_delta_max*D , N)
w_1, Delta_fig_1 = np.meshgrid(w_1, Delta_fig_1)
U = np.sqrt((Re_Im(El_coef, D+Delta_fig_1, D, K, Γ, Rabi, w_1)[0])**2 + (Re_Im(El_coef, D+Delta_fig_1, D, K, Γ, Rabi, w_1)[1])**2)
axs[0].imshow(U.T, cmap='jet', origin='lower', extent=[(D+Delta_fig_1.min())/D, (D+Delta_fig_1.max())/D, w_1.min()/D,w_1.max()/D],aspect='auto')
axs[0].set_xlabel(r"$\omega_{C}$ [$\Delta$]")
axs[0].set_ylabel(r" $\omega_{L}$ [$\Delta$]")
#plt.colorbar(label=r"Amplitude |q($\omega_{L}$, $\delta$)|     [A.U]")
axs[0].text(0.97, 1.31 , r'$\omega$+', fontsize=18, color = "white", fontweight='bold')
axs[0].text(1, 0.84 , r'$\omega$-', fontsize=18, color = "white", fontweight='bold')
axs[0].text(0.58, 1.34 , '(a)', fontsize=12, color = "white", fontweight='bold')

axs[0].plot(line_1, splitting_visual, color="white", linewidth=2)
axs[0].annotate('', xy=(line_1[1], splitting_visual[1]-0.01), xytext=(line_1[0], splitting_visual[0]-0.01), arrowprops=dict(arrowstyle='<-', color='white', lw=2))
axs[0].annotate('', xy=(line_1[-2], splitting_visual[-2]+0.01),  xytext=(line_1[-1], splitting_visual[-1]+0.01), arrowprops=dict(arrowstyle='<-', color='white', lw=2))
axs[0].text(1.02, 1.1 , r"$\Omega_{R}$", fontsize=13.5, color = "white")
axs[0].set_aspect('equal')
axs[0].legend()
#Γo show the actuals parameters
'''
parameters = [] 
for i in range(2):
    parameters.append(f"(Quality factor) Q = {Q}\n(Non radiative emissions coefficient) Γ = {Γ/wc:.2e}[wc]\n(Rabi frequency) Ω = {Rabi/wc:.2}% [wc]\n(Detuning) $\delta$ = {Delta/wc:.2}[wc])
axs_list = [axs[0], axs[1]]
axs_list = [axs[0], axs[1]]
for ax, text in zip(axs_list, parameters):
    ax.text(0, -0.14, text,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=7.5,
            bbox=dict(facecolor='white', alpha=0.3))
'''
'''
parameters = f"Q = {Q}\nΓ = {Γ/wc:.2e}[wc]\n$\Omega$ = {Rabi/wc:.2}% [wc]\n$\delta$ = {Delta/wc:.2}[wc])"
axs[0].text(0.02, -0.28, parameters,
            transform=axs[0].transAxes,
            ha='left', va='top',
            fontsize=9.8,
            bbox=dict(facecolor='white', alpha=0))    
'''

'''
# Figure 3: Energy by w and Δ.
fig = plt.figure('|q(w, Δ)|')
N = 300
Delta_plot = wc
w_1 = np.linspace(0.2*Delta_plot, 1.8*Delta_plot, N)
Wc_fig_1 = np.linspace(0.7*Delta_plot, 1.3*Delta_plot , N)
w_1, W_fig_1 = np.meshgrid(w_1, Wc_fig_1)
U = np.sqrt((Re_Im(El_coef, Wc_fig_1, Delta_plot, K, Γ, Rabi, w_1)[0])**2 + (Re_Im(El_coef, Wc_fig_1, Delta_plot, K, Γ, Rabi, w_1)[1])**2)
plt.imshow(U.T, cmap='magma', origin='lower', extent=[Wc_fig_1.min()/Delta_plot , Wc_fig_1.max()/Delta_plot , w_1.min()/Delta_plot ,w_1.max()/Delta_plot] ,aspect='auto')
plt.xlabel(r"$\omega_{C}$ [$\Delta$]")
plt.ylabel(r" $\omega_{L}$ [$\Delta$]")
plt.colorbar(label=r"Amplitude |q($\omega_{L}$, $\Delta$)|     [$\Delta$]")
plt.text(0, 1.31 , r'$\omega$+', fontsize=18, color = "white", fontweight='bold')
plt.text(0, 0.84 , r'$\omega$-', fontsize=18, color = "white", fontweight='bold')
plt.legend()
'''
'''
# Figure 3: Energy by w and Δ.
fig = plt.figure(figsize=(5,5), constrained_layout=True)
fact_delta_max, fact_W_max = 0.5, 1.6
N = 300

w_1 = np.linspace(0.4*D, fact_W_max*D, N)
Delta_fig_1 = np.linspace(-fact_delta_max*D, fact_delta_max*D , N)

w_1, Delta_fig_1 = np.meshgrid(w_1, Delta_fig_1)

U = np.sqrt((Re_Im(El_coef, D+Delta_fig_1, D, K, Γ, Rabi, w_1)[0])**2 + (Re_Im(El_coef, D+Delta_fig_1, D, K, Γ, Rabi, w_1)[1])**2)

plt.imshow(U.T, cmap='jet', origin='lower', extent=[(D+Delta_fig_1.min())/D, (D+Delta_fig_1.max())/D, w_1.min()/D,w_1.max()/D],aspect='auto')

plt.xlabel(r"$\omega_{C}$ [$\Delta$]")
plt.ylabel(r" $\omega_{L}$ [$\Delta$]")
#plt.colorbar(label=r"Amplitude |q($\omega_{L}$, $\delta$)|     [A.U]")
plt.text(0.97, 1.31 , r'$\omega$+', fontsize=18, color = "white", fontweight='bold')
plt.text(1, 0.84 , r'$\omega$-', fontsize=18, color = "white", fontweight='bold')
plt.legend()
'''





fig, ax = plt.subplots(2,1,figsize=(5, 3), constrained_layout=True)
ax[0].plot(w , Re_im_q(wc, w, K, El_coef)[0], color="grey")
ax[0].plot(w , analytic_simplification_re_im(wc, w, K, El_coef)[0], "r--", color="black")
ax[0].set_xlabel(r"$\omega_{L}$ [$\omega_{C}$]")
ax[0].set_ylabel(r"Re[q(w)] [A.U]")
ax[0].legend()
ax[1].plot(w , Re_im_q(wc, w, K, El_coef)[1], color="grey")
ax[1].plot(w , analytic_simplification_re_im(wc, w, K, El_coef)[1],"r--", color="black")
ax[1].set_xlabel(r"$\omega_{L}$ [$\omega_{C}$]")
ax[1].set_ylabel(r"Im[q(w)] [A.U]")
ax[1].legend()
#ax[0,1].plot(w , Re_im_q(wc, w, K, El_coef)[0], color="grey")






#================Aproximation window ==================================

fig, ax = plt.subplots(2,1,figsize=(5, 7), constrained_layout=True)
X_MAX = 0.1*wc
N =300
W_laser = np.linspace(1*wc, 1.1*wc , N)
Rabi_line = np.linspace(0, ((K-Γ)/4),500)
Rabi_line_sc = np.linspace(((K-Γ)/4), X_MAX,500)
Rabi_w_wc = np.linspace(0,((K-Γ)/4), 200)
Rabi_line_ge = np.linspace(0, X_MAX , N)


#print("the solution is", solve_deg_4(wc, wc, K, Γ, 0))
list_w1, list_w2, list_w1_sc, list_w2_sc, list_w1_G, list_w2_G, list_w1_sc_G, list_w2_sc_G = analytic_resolution_w_rabi(Rabi_line, Rabi_line_sc, wc, K, Γ)
#=========== eigenfrequencies vs Rabi frequency ========================

ax[0].plot(Rabi_line/wc, list_w1,"r--", color="grey", linewidth = 2.5,label = "$\omega$$\pm$ numeric"  )
ax[0].plot(Rabi_line/wc, list_w2,"r--", color="grey", linewidth = 2.5)
ax[0].plot(Rabi_line_sc/wc, list_w1_sc,"r--", color="grey", linewidth = 2.5  )
ax[0].plot(Rabi_line_sc/wc, list_w2_sc,"r--", color="grey", linewidth = 2.5)


ax[0].plot(Rabi_line_sc/wc , strong_coupling_w(wc, K, Γ, Rabi_line_sc)[0]/wc, color="black", label = r"$\omega$$\pm$" )
ax[0].plot(Rabi_line_sc/wc , strong_coupling_w(wc, K, Γ, Rabi_line_sc)[1]/wc, color="black")
ax[0].plot(Rabi_w_wc , weak_coupling_w(wc, K, Γ, Rabi_w_wc)[0]/wc, color="black")

#imshow ??
fact_W_min, fact_W_max = weak_coupling_w(wc, K, Γ, 0)[0], strong_coupling_w(wc, K, Γ, X_MAX)[0]
N = 300
w_1 = np.linspace(fact_W_min*wc, fact_W_max*wc, N)
w_1, Rabi_line_ge = np.meshgrid(w_1, Rabi_line_ge)
U = np.sqrt((Re_Im(El_coef, wc, wc, K, Γ, Rabi_line_ge, w_1)[0])**2 + (Re_Im(El_coef, wc, wc, K, Γ, Rabi_line_ge, w_1)[1])**2)
#ax[0].imshow(U.T, cmap='jet', origin='lower', extent=[Rabi_line_ge.min()/wc, Rabi_line_ge.max()/wc, w_1.min()/wc,w_1.max()/wc],aspect='auto')
#zz


ax[0].set_xlabel(r"$\Omega$ [$\omega_{C}$]")
ax[0].set_ylabel(r"$\omega_{\pm}$ [$\omega_{C}$]" )
ax[0].axvline(x=((K-Γ)/4)/wc,color='black',linestyle=':')
ax[0].text( max(Rabi_line)*0.3/wc, 1.045, 'W.C', fontsize=13, color = "black", fontweight='bold')
ax[0].text( max(Rabi_line)*0.07/wc, 1.04, '(Purcell Effect)', fontsize=9, color = "black")
ax[0].text( max(Rabi_line)*1.35/wc, strong_coupling_w(wc, K, Γ, max(Rabi_line)*2)[0]/wc, 'S.C', fontsize=13, color = "black", fontweight='bold')
ax[0].text( 0.07, 1.082, 'P+', fontsize=13, color = "black")
ax[0].text( 0.07, 1.01, 'P-', fontsize=13, color = "black")
ax[0].text( max(Rabi_line)*1.1/wc, 1.08, r'$\Omega>(\frac{K-Γ}{4})$', fontsize=14, color = "black", fontweight='bold')
ax[0].text(0.016,1.09, '(a)', fontsize=12, color='black', fontweight='bold')
ax[0].set_xlim(0, X_MAX/wc)
ax[0].set_aspect('auto')
ax[0].legend()


#=======Emission vs Rabi frequency =====================

ax[1].plot(Rabi_line/wc, list_w1_G,"r--", color="grey" , linewidth = 2.5, label = r"Γ$\pm$ numeric " )
ax[1].plot(Rabi_line/wc, list_w2_G,"r--", color="grey", linewidth = 2.5)
ax[1].plot(Rabi_line_sc/wc, list_w1_sc_G,"r--", color="grey" , linewidth = 2.5 )
ax[1].plot(Rabi_line_sc/wc, list_w2_sc_G,"r--", color="grey", linewidth = 2.5)

ax[1].hlines( strong_coupling_w(wc, K, Γ, 1)[2]/wc, (K-Γ)/4/wc, (K-Γ)/4/wc+X_MAX-(K-Γ)/4/wc, color = "black")
#ax[1].plot(Rabi_Γ_sc/wc , strong_coupling_w(wc, K, Γ, Rabi_Γ_sc)[2]/wc, "r--", color="black",label = "$\Gamma$ strong coupling")
ax[1].plot((Rabi_line)/wc , weak_coupling_w(wc, K, Γ, Rabi_line)[1]/wc, color="black",label = "Γ$\pm$")
ax[1].plot((Rabi_line)/wc , weak_coupling_w(wc, K, Γ, Rabi_line)[2]/wc, color="black")
ax[1].axvline(x=((K-Γ)/4)/wc,color='black',linestyle=':')
ax[1].set_xlabel(r"$\Omega$ [$\omega_{C}$]")
ax[1].set_ylabel(r"$\Gamma_{\pm}$ [$\omega_{C}$]")
ax[1].set_xlim(0, X_MAX/wc)
#ax[1].text( max(Rabi_line)*1.1/wc, 0.007, r'$\Omega_{RA}>(\frac{K-Γ}{4})$', fontsize=14, color = "black", fontweight='bold')
ax[1].text( max(Rabi_line)*0.3/wc, 0.0125, 'W.C', fontsize=13, color = "black", fontweight='bold')
ax[1].text( max(Rabi_line)*0.07/wc, 0.0114, '(Purcell Effect)', fontsize=9, color = "black")
ax[1].text( max(Rabi_line)*1.6/wc, 0.007, 'S.C', fontsize=13, color = "black", fontweight='bold')
ax[1].text(0.016,0.023, '(b)', fontsize=12, color='black', fontweight='bold')
ax[1].set_aspect('auto')
ax[1].legend()


print(splitting*3.34e15*h_bar/e_v)

plt.show()