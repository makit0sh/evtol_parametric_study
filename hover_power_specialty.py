# -*- coding: utf-8 -*-


## modules -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


## parameters ----------------------------------------------------------------------------------------------------------
FAN_CONF = {
        'WTO': 600,
        'Omega': 1500,
        'n': 18,
        'rho': 1.225,
        'b': 12,
        'R': 0.4,
        'c': 0.04,
        'theta_0': 60,
        'theta_t': 20,
        'a': 5.73,
        'Cd': 0.0125,
        'B': 0.97,
        }


## functions -----------------------------------------------------------------------------------------------------------
def hover_fan_regulate_r(conf):
    """ホバリング時のファンの挙動，ファン半径を最適化する
    """
    # ファン全体が出す必要がある推力
    T_need_all = conf['WTO'] * 9.8
    T_need = T_need_all / conf['n']

    # 角速度
    Omega = conf['Omega']  # rpm
    omega = Omega / 60 * 2 * np.pi      # rad

    # 半径
    R = np.arange(0.1, 0.5, 0.0001)

    # m^2, ローター円盤面積
    S = np.pi * R**2
    # ソリディティ, ヘリコプタ入門 4.24
    sigma = conf['b'] * conf['c'] / np.pi / R

    # m/s, 吹き下ろし速度 (ホバリング時), ヘリコプタ入門 4.1
    v_0 = np.sqrt(T_need/2/conf['rho']/S)
    # m/s, 翼端の速度
    v_t = R * omega
    # N/m^2, 動圧
    q = conf['rho'] * S * v_t**2

    # 流入比, ヘリコプタ入門 5.8
    Lambda = v_0/omega/R
    # rad, 吹き下ろし角
    phi_t = -np.arctan(v_0/omega/R)

    # 推力係数, 翼素理論, 航空工学I 4.127a
    C_T_guess = conf['a']*sigma/2*(conf['B']**3 *np.deg2rad(conf['theta_0'])/3 - conf['B']**4 *np.deg2rad(conf['theta_t'])/4 - conf['B']**2 * Lambda/2) - sigma*conf['Cd']*Lambda/4
    T_guess = q * C_T_guess

    err = T_guess - T_need
    err_min_idx = np.argmax(1/err)
    #idx = np.where(err/T_need < 0.005)
    #print(idx)

    R = R[err_min_idx]
    sigma = sigma[err_min_idx]
    v_t = v_t[err_min_idx]
    q = q[err_min_idx]
    Lambda = Lambda[err_min_idx]
    phi_t = phi_t[err_min_idx]
    C_T_guess = C_T_guess[err_min_idx]
    T_guess = T_guess[err_min_idx]
    C_T = C_T_guess

    # N, 推力, ヘリコプタ入門 4.22
    T = q * C_T
    # トルク係数, パワ係数と同じ, ヘリコプタ入門 4.39
    C_Q = -phi_t*C_T + sigma*conf['Cd']/8
    # Nm, トルク, ヘリコプタ入門 4.36
    Q = q * R * C_Q
    # W, パワー
    P = Q * omega
    return {'WTO': conf['WTO'],
            'n': conf['n'],
            'Omega': Omega,
            'R': R,
            'T': T,
            'Q': Q,
            'P': P,
            'P_total': P*conf['n'],
            'C_T': C_T}



def hover_fan_specialty():
    wto_list = [300, 400, 500, 600]
    Omega_list = [1500, 2000, 4000, 6000, 8000]
    n_fan_list = [10, 12, 16, 18, 20, 22, 24]
    color_list = ["green", "gold", "royalblue", "orangered"]

    fan_conf = dict(FAN_CONF)
    for i, wto in enumerate(wto_list):
        fan_conf['WTO'] = wto
        color = color_list[i]
        for j, Omega in enumerate(Omega_list):
            fan_conf['Omega'] = Omega
            R_list_temp = []
            P_list_temp = []
            for k, n in enumerate(n_fan_list):
                fan_conf['n'] = n
                result = hover_fan_regulate_r(fan_conf)
                R = result['R']
                P_one_fan = result['P']
                P_all = P_one_fan * 1e-3 * n
                R_list_temp.append(R)
                P_list_temp.append(P_all)
            R = np.array(R_list_temp)
            P = np.array(P_list_temp)
            plt.plot(R, P, #color=color, 
                    color = cm.hsv(1/len(wto_list)*i+0.1/len(Omega_list)*j+0.01/len(n_fan_list)*k),
                    marker="o", linestyle="solid", label="WTO={}, Omega={}, n={}".format(wto, Omega, n))
    plt.grid()
    plt.hlines([100], 0, 0.5)
    plt.title("power tokusei when hovering")
    plt.xlabel("fan radius [m]")
    plt.ylabel("gross power [kW]")
    #plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.subplots_adjust(right=0.65)
    plt.savefig("hover_specialty_sample.png")
    plt.show()


def main():
    print(hover_fan_regulate_r(FAN_CONF))
    hover_fan_specialty()
    return


## execution -----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
