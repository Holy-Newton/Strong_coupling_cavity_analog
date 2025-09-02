import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.signal import find_peaks
import math

q_0, q_p0 = 1, 0
Q_0, Q_p0 = 0, 0
wc = 1 #Cavity frequency
Delta = wc*0.00
D = wc + Delta  #Emitter frequency, "diffusion"
Q =  10#quality[10;100]
K = (wc)/Q  #"cavity diffusion"
T = Delta/(wc/25e-12)
E0, Rabi = 0.1, 0.32*wc# Force of the drive # (/ mass) , #Rabi frequency
Γ = K * 5e-4

wl = wc*1.2 # laser frequency for temporal representation 
time = 40*2*np.pi/wc
w, t = np.linspace(0.5*wc, 2*wc, 400), np.linspace(0, time, 500) #laser frequency # time for temporal representation
Delta_line = np.linspace(0, 8*Delta, 300)

h_bar = 1.05457e-34
e_v = 1.602e-19
#==========PARAMETERS OF THE HOMOGEN SOLUTION==========
def solve_deg_4(wc, D, K, T, Rabi):  #ancienne tentative: retourne les bonnes valeurs de 
    A = np.array([[K/2, 0], [0, T/2]])

    B = np.array([[wc**2 + wc*Rabi, -wc*Rabi],
                [-wc*Rabi, D**2 + wc*Rabi]])

    M =np.block([[np.zeros((2,2)), np.identity(2)],
                [-B, -A]])
    racines = np.linalg.eigvals(M)    
    list =[]
    for i,r in enumerate(racines):
        list.append(r)
    z1, z2, z3, z4 = list
    for i in list:
        print(i)
    #return z1, z2, z3, z4
#trouver les valeurs propres et vecteurs propres des deux pendules + influence de l'une sur l'autre
def eigen_space(wc, D, K, T, Rabi):
    matrice = np.array([[0, 1, 0, 0],
                       [-wc**2-wc*Rabi, -K/2, wc*Rabi, 0],
                       [0, 0, 0, 1],
                       [wc*Rabi, 0, -D**2-wc*Rabi, -T/2]])
    eigenvalues, eigenvectors = np.linalg.eig(matrice)
    print("===========Valeurs propres==============")
    print(eigenvalues)
    print("==============Vecteurs propres=========")
    for i in eigenvectors:
        print(" ")
        print(i)
    return eigenvalues, eigenvectors
#==========PARAMETERS OF THE PARTICULAR SOLUTION==========
# return re and im parts for each oscillator
def Re_Im(E0, wc, D, K, T, Rabi, wl ):
    a = wc**2 + Rabi*wc - wl**2
    b = wl*K/2
    a_p = D**2 + Rabi*wc - wl**2
    b_p = wl*T/2
    gamma = -wc*Rabi
    alpha = a*a_p - b*b_p - gamma**2
    beta = b_p*a + b*a_p

    req = E0*( a_p*alpha + b_p* beta)/ (alpha **2 + beta**2)
    imq = E0*( b_p*alpha - a_p* beta)/ (alpha **2 + beta**2)

    reQ = E0*(-gamma * alpha)/ (alpha **2 + beta**2)
    imQ = E0*(gamma * beta)/ (alpha **2 + beta**2)

    return req, imq, reQ, imQ

#return the amplitudes of the drive particular solution 
def q_Q_w(E0, wc, D, K, T, Rabi, wl ):

    req, imq, reQ, imQ = Re_Im(E0, wc, D, K, T, Rabi, wl )
    return np.sqrt(req**2 + imq**2), np.sqrt(reQ**2 + imQ**2)

#return the phase of each oscillator
def dephasage(E0, wc, D, K, T, Rabi, wl ):

    req, imq, reQ, imQ = Re_Im(E0, wc, D, K, T, Rabi, wl )
    
    return np.angle(req + 1j * imq), np.angle(reQ + 1j * imQ)

#========== FINAL SOLUTION --> HOMO + PARTICULAR SOLUTIONS =================
def q_Q(t, E0, wl, wc, D, K, T, Rabi, q_0, q_p0, Q_0, Q_p0):

    #particular solution
    A_q, A_Q = q_Q_w(E0, wc, D, K, T, Rabi, wl )
    phi_q, phi_Q = dephasage(E0, wc, D, K, T, Rabi, wl )
    q_p = A_q * np.cos(wl * t + phi_q)
    Q_p = A_Q * np.cos(wl * t + phi_Q)
    #homogeneous solution
 
    eigenvalues, eigenvectors = eigen_space(wc, D, K, T, Rabi)



    #to have the particular init conditions for the calculus of the coefficients of each eigen vectors
    q_correction = q_0 - A_q*np.cos(phi_q)
    Q_correction = Q_0 - A_Q*np.cos(phi_Q)
    dq_correction = q_p0 + A_q*wl*np.sin(phi_q)
    dQ_correction = Q_p0 + A_Q*wl*np.sin(phi_Q)
    init_coefficient = [q_correction, dq_correction, Q_correction, dQ_correction]

    '''Linear combinaison: I want q(0)- q_correctino = sum_i_to_N(Ci*Vi*exp(Zi * 0)) and same for q'(0)
    So vector * exp(value*0)* coef  = condition(0)    ----->  coef = vector-1 * condition(0) '''
    sol = np.linalg.solve(eigenvectors, init_coefficient)  #caution vector before lol <==> matricial operation 
    print("==========solution init_coeff==============")
    print(sol)
    q_h = Q_h = 0
    # definition of each member of the linear combinaison of q: sum of i to N of ci*vqi*exp(zi *t)
    for i in range(4):
        q_h += sol[i]*eigenvectors[0,i]*np.exp(eigenvalues[i]*t)
        Q_h += sol[i]*eigenvectors[2,i]*np.exp(eigenvalues[i]*t)

    # homogeneous + particular solution
    q = q_h + q_p
    Q = Q_h + Q_p
    return q, Q




#==================== numeric resolution ========================
def fun_ivp(t, z, E0, wc, D, K, T, Rabi, wl):
    q, dq_dt, Q, dQ_dt = z
    q_drive = E0 * np.cos(wl * t)
    Q_drive = 0
    dq_2dt = -(wc**2 +wc*Rabi)*q -(K/2) * dq_dt + wc*Rabi* Q + q_drive
    dQ_2dt = -(D**2 +wc*Rabi)*Q -(T/2) * dQ_dt + wc*Rabi * q + Q_drive
    return [dq_dt, dq_2dt, dQ_dt, dQ_2dt]

def rabi_splitting(E0, wc, D, K, T, Rabi, wl):
    intensities = []
    for i in wl:
        re, im, reQ, imQ = Re_Im(E0, wc, D, K, T, Rabi, i)
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
def Re_im_q(wc, w, K, E0):
    w0 = w
    gamma = ((wc**2-w0**2)**2+(K*w0/2)**2)
    re = E0*(wc**2-w0**2)/gamma
    im = E0*(-K*w0/2)/gamma   ### mais si utilisation dxse exp(-iwt): alors pic de résonnance im positif
    return re,im














#=========représentation graphique=========

# Fenêtre 1 : Analyse fréquentielle
fig, axs = plt.subplots(2,1,figsize=(7, 9), constrained_layout=True)
#Partie réelle
axs[0].plot(w/wc, Re_Im(E0, wc, D, K, T, Rabi, w )[0], color='black', label = "Re[q(w)]")
axs[0].plot(w/wc, Re_Im(E0, wc, D, K, T, Rabi, w )[1], color='orange', label = "Im[q(w)]")
axs[0].set_title("Real and imaginary parts of q_part(w)")
axs[0].set_ylabel("Amplitude [U.A]")
axs[0].set_xlabel("[w/wc]")
axs[0].grid(True)
axs[0].legend()

# Partie imaginaire
axs[1].plot(w/wc, Re_Im(E0, wc, D, K, T, Rabi, w )[2], color='black', label = "Re[Q(w)]")
axs[1].plot(w/wc, Re_Im(E0, wc, D, K, T, Rabi, w )[3], color='orange',label = "Im[Q(w)]")
axs[1].set_title("Real and imaginary parts of Q_part(w)")
axs[1].set_ylabel("Amplitude [U.A]")
axs[1].set_xlabel("[w/wc]")
axs[1].grid(True)
axs[1].legend()

parameters = []
for i in range(2):
    parameters.append(f"Q = {Q}\nT = {T:.2e}\nΩ = {Rabi/wc:.2}% [wc]  \nWlaser = {wl/wc:.2e} [wc]")
axs_list = [axs[0], axs[1]]
for ax, text in zip(axs_list, parameters):
    ax.text(0.9, 0.4, text,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.5))








fig, axs = plt.subplots(2,1,figsize=(7, 7), constrained_layout=True)
# Phase
axs[0].plot(w/wc, np.unwrap(dephasage(E0, wc, D, K, T, Rabi, w )[0]), color='blue', label = "Phi[q(w)]")
axs[0].plot(w/wc, np.unwrap(dephasage(E0, wc, D, K, T, Rabi, w )[1]), color='grey', label = "Phi[Q(w)]")
axs[0].set_title("Phase shift of q_part(w)")
axs[0].set_ylabel("Phase shift [rad]")
axs[0].set_xlabel("[wlaser/wc]")
axs[0].grid(True)
axs[0].legend()
# Intensité

splitting, w1, w2, max_val = rabi_splitting(E0, wc, D, K, T, Rabi, w ) 
splitting_visual = np.linspace(w1, w2, 30)
splitting_points = []
for i in splitting_visual:
    splitting_points.append(max_val/2)

axs[1].plot(w/wc, q_Q_w(E0, wc, D, K, T, Rabi, w )[0]**2, color='blue', label = "I[q(w)]")
#for i in range(20):
#    u = -i*Delta
#    axs[1].plot(w/wc, q_Q_w(E0, wc, D+u, K, T, Rabi, w )[0]**2, color='grey')
#axs[1].plot(w/wc, q_Q_w(E0, wc, D, K, T, Rabi, w )[1]**2, color='grey', label = "I[Q(w)]")
axs[1].plot(w/wc, Re_im_q(D, w, K, E0)[0]**2+Re_im_q(D, w, K, E0)[1]**2,"r--" ,color='red', label = "Emitter intensity")
axs[1].plot(splitting_visual/wc, splitting_points, "r--" ,color='black', label = f"Rabi splitting:{splitting*h_bar*1000/e_v:.3f}meV",linewidth=3.5)
axs[1].plot(w/wc, )
axs[1].set_title("Intensity of q_part(w) Drived by a wl laser frequency")
axs[1].set_ylabel("I(w) [U.A]")
axs[1].set_xlabel("[wlaser/wc]")
axs[1].legend()
axs[1].grid(True)

#To show the actuals parameters
axs_list = [axs[0], axs[1]]
for ax, text in zip(axs_list, parameters):
    ax.text(0.9, 0.4, text,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.5))
    









# Fenêtre 3 : q((t) discriminant
#solution of the analytic resolution
solution = solve_ivp(fun_ivp, [0, time], [q_0, q_p0, Q_0, Q_p0], method='RK45', max_step=0.1, args = (E0, wc, D, K, T, Rabi, wl))

fig, axs = plt.subplots(2,1,figsize=(11, 7), constrained_layout=True)
    
axs[0].plot(solution.t/(2*np.pi)*wc, solution.y[0], "r--",linewidth=4, label="Résolution numérique q(t)", color="grey")
axs[0].plot(solution.t/(2*np.pi)*wc, q_Q(solution.t,E0, wl, wc, D, K, T, Rabi, q_0, q_p0, Q_0, Q_p0)[0], linewidth=2, label="q(t) analytique", color="red")
axs[0].set_title("q(t) numeric and analytic resolutions")
axs[0].set_ylabel("q(t) [U.A]")
axs[0].set_xlabel("periods [2π / wc]")
axs[0].grid(True)
axs[0].legend()
# Partie imaginaire
axs[1].plot(solution.t/(2*np.pi)*wc, solution.y[2], "r--",linewidth=4, label="Résolution numérique Q(t)", color="green")
axs[1].plot(solution.t/(2*np.pi)*wc, q_Q(solution.t,E0, wl, wc, D, K, T, Rabi, q_0, q_p0, Q_0, Q_p0)[1], linewidth=2, label="Q(t) analytique", color="blue")
axs[1].set_title("Q(t) numeric and analytic resolutions")
axs[1].set_ylabel("Q(t) [U.A]")
axs[1].set_xlabel("periods [2π / wc]")
axs[1].grid(True)
axs[1].legend()

fig1 = plt.figure(figsize=(7, 3))
plt.plot(solution.t/(2*np.pi)*wc, solution.y[0], ":",linewidth=2.8, label="q(T) numeric", color="grey")
plt.plot(solution.t/(2*np.pi)*wc, q_Q(solution.t,E0, wl, wc, D, K, Γ, Rabi, q_0, q_p0, Q_0, Q_p0)[0], linewidth=1.6, label="q(T) analytic", color="black")
plt.ylabel("q(T) [U.A]")
plt.xlabel("T [2π / wc]")
plt.legend()

parameters = []
for i in range(2):
    parameters.append(f"K = {K:.2e}\nT = {T:.2e}\nΩ = {Rabi:.2e}\nWlaser = {wl/wc:.2e} [wc]\nWc = {wc:.2e} [A.U]")
axs_list = [axs[0], axs[1]]
for ax, text in zip(axs_list, parameters):
    ax.text(0.9, 0.4, text,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.5))
    

"""
fig = plt.figure("Absorbance")
wc = np.linspace(0, 1.2*wc, 300)
for i in range(10):
    u = i*0.1*wc
    plt.show(w, Re_Im(E0, wc, D, K, T, Rabi, w+u )[0]**2 + Re_Im(E0, wc, D, K, T, Rabi, w+u )[1]**2)
    
"""











    

# Figure 3: Energy by w and Δ.
fig = plt.figure('|q(w, Δ)|')
fact_delta_max, fact_W_max = 2, 2
N = 300
w_1 = np.linspace(0, fact_W_max*wc, N)
Delta_fig_1 = np.linspace(0*wc, fact_delta_max*wc , N)
w_1, Delta_fig_1 = np.meshgrid(w_1, Delta_fig_1)
U = np.sqrt((Re_Im(E0, wc, Delta_fig_1, K, T, Rabi, w_1)[0])**2 + (Re_Im(E0, wc, Delta_fig_1, K, T, Rabi, w_1)[1])**2)
plt.imshow(U.T, cmap='magma', origin='lower', extent=[Delta_fig_1.min(), Delta_fig_1.max(), w_1.min(),w_1.max()],aspect='auto')
plt.title(r"$|q(\omega, \delta)|$")
plt.xlabel(r"$\delta$ / $\omega$c")
plt.ylabel(r"$\Delta$ / $\omega$c")
plt.colorbar(label=r"$|q(w, Δ)|^2 [A.U]$")




'''
# Figure 4:.
fig = plt.figure('|q(w, Δ)|^2')
for i in range(10):
    plt.plot()
plt.title(r"$|q(\omega, \delta)|$")
plt.xlabel(r"$\delta$ / $\omega$c")
plt.ylabel("eV")
plt.colorbar(label=r"$|q(w, Δ)| [A.U]$")

'''

# Figure: 4 plot 3d
fig = plt.figure('|q(w, Δ)|^2')
Delta_3Dplot = np.linspace(-1*D, D, 200)
W, DELTA = np.meshgrid(w, Delta_3Dplot)
INTENSITY = np.zeros_like(W)

for i in range(DELTA.shape[0]):
    for j in range(W.shape[1]):
        req, imq, _, _ = Re_Im(E0, wc, wc + DELTA[i, j], K, DELTA[i, j]/10, Rabi, W[i, j])
        INTENSITY[i, j] = np.sqrt(req**2 + imq**2)

ax3d = fig.add_subplot(111, projection='3d')
ax3d.plot_surface(W, DELTA, INTENSITY, cmap='viridis')

ax3d.set_title("Diagramme 3D : |q(w, Δ)|^2")
ax3d.set_xlabel("w")
ax3d.set_ylabel("Δ")
ax3d.set_zlabel("|q(w, Δ)|")


plt.show()