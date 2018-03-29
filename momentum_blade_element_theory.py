# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def cruise_fan(conf, V, omega):
    # ソリディティ
    sigma = conf['b']*conf['c']/np.pi/conf['R']
    # m, プロペラ直径
    D = 2*conf['R']

    # 積分用の無次元座標
    dx = 0.01 # 積分間隔
    N_tmp = int((0.9-0.15)/dx)
    x = np.linspace(0.15, 0.9, N_tmp)
    beta = np.deg2rad(np.linspace(conf['theta_0'], conf['theta_t'], N_tmp))

    r = x*conf['R']
    phi = np.arctan(V/r/omega)
    # プラントルの翼端損失
    F = 2/np.pi*np.arccos(np.exp(-conf['b']*(1-x)/2/np.sin(beta[-1])))
    # 翼型の情報 (かなり荒い近似)
    m0 = 2*np.pi
    Cd = 0.02
    alpha0 = 0
    X = np.tan(phi) + sigma*m0/8/x/F/np.cos(phi)
    Y = sigma*m0/8/x/F/np.cos(phi)*(beta-phi-alpha0)
    # プラントルの誘導迎角
    alpha_i = (-X+np.sqrt(X**2 + 4*Y)) / 2
    #alpha_i = (beta-phi-alpha0)/(1+8*x*np.sin(phi)/sigma/m0)
    Cl = m0 * (beta-phi-alpha0-alpha_i)
    dCT = sigma*np.pi**3/8 *(np.cos(alpha_i))**2/(np.cos(phi))**2 * x**2 * (Cl*np.cos(phi+alpha_i) - Cd*np.sin(phi+alpha_i)) * dx
    dCQ = sigma*np.pi**3/16 *(np.cos(alpha_i))**2/(np.cos(phi))**2 * x**3 * (Cl*np.sin(phi+alpha_i) + Cd*np.cos(phi+alpha_i)) * dx

    # 区間ではピッチ一定として積分
    CT = np.sum(dCT)
    CQ = np.sum(dCQ)
    CP = 2*np.pi*CQ

    # 1/s, 単位時間あたり回転数
    n = omega/2/np.pi
    # プロペラ進行率
    J = V / n / D
    # プロペラ効率
    eta = J * CT / CP

    T = CT * conf['rho']*n**2*D**4
    Q = CQ * conf['rho']*n**2*D**5

    return {
            'J': J,
            'T': T,
            'Q': Q,
            'CT': CT,
            'CQ': CQ,
            'CP': CP,
            'eta': eta,
            }

def main():
    REAR_FAN_CONF = {
        'loading_ratio': 2/3,
        'rho': 1.225,
        'n': 12,
        'b': 12,
        'R': 0.3,
        'c': 0.04,
        'theta_0': 60,
        'theta_t': 20,
        'a': 5.73,
        'Cd': 0.0125,
        'B': 0.97,
        }

    # 8000 rpm でテスト
    print( cruise_fan(REAR_FAN_CONF, V=100/3.6, omega=8000*np.pi/30) )

    V_list = np.linspace(10, 300)/3.6
    results = [ cruise_fan(REAR_FAN_CONF, V, omega=8000*np.pi/30) for V in V_list ]
    plt.plot([ result['J'] for result in results ], [ result['eta'] for result in results ])
    plt.show()

    return

if __name__ == '__main__':
    main()
