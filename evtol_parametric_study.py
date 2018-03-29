# -*- coding: utf-8 -*-


## modules -------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

from copy import copy

import momentum_blade_element_theory

## parameters ----------------------------------------------------------------------------------------------------------
CONF = {
        # 概要
        #'passenger': 2, # 乗員数
        #'payload': 200, # kg, 搭載重量
        'MTO': 600,  # kg, 最大離陸重量
        'V_c': 307 / 3.6,  # m/s, 巡航速度
        'z': 0,  # m, 飛行高度
        #'Range': 250,  # km
        # 重心
        'x_cg': 0,  # m, 重心（主翼のc/4の前方）
        'l_t': 4.44,  # m, 主翼c/4から先尾翼c/4までの距離
        'l_ffan': 3.41,  # m, 前ファンと主翼c/4との距離
        'l_rfan': 1.71,  # m, 後ファンと主翼c/4との距離
        # 主翼
        'b': 5.6,  # m, 翼幅
        'c': 0.8,  # m, 翼弦長
        'e': 0.9,  # 翼効率
        # 複葉
        'is_biplane': True,  # 複葉かどうか
        'h': 0.75,  # 複葉のギャップ
        # 先尾翼
        'b_t': 4.0,  # m, 先尾翼の翼幅
        'c_t': 0.7,  # m, 先尾翼の翼弦長
        'i_t': 0.437,  # deg, 先尾翼の取り付け角
        'e_t': 0.9,  # 先尾翼の翼効率
        # 胴体
        'b_f': 1.15,  # m, 胴体幅
        'L': 7.71,  # m, 胴体長
        'is_streamline': True,  # 胴体が流線型かどうか
        # 前ファン
        'b_ffan': 2.25,  # m, 前ファン幅
        'c_ffan': 0.51,  # m, 前ファン弦長
        'h_ffan': 0.62,  # m, 前ファン高さ
        # 後ファン
        'b_rfan': 2.25,  # m, 後ファン幅
        'c_rfan': 1.02,  # m, 後ファン弦長
        'h_rfan': 1.11,  # m, 後ファン高さ
    }

FRONT_FAN_CONF = {
        'loading_ratio': 1/3,
        'rho': 1.225,
        'n': 6,
        'b': 12,
        'R': 0.3,
        'c': 0.04,
        'theta_0': 60,
        'theta_t': 20,
        'a': 5.73,
        'Cd': 0.0125,
        'B': 0.97,
        }

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



## functions -----------------------------------------------------------------------------------------------------------
# 空気の物性
def air(z):
    """大気条件を算出する
    Args:
        z (float): m, 飛行高度
    Returns:
        (dict): 大気の条件
    """
    p0 = 1013.0
    t0 = 15
    p = p0 * np.power(1-0.0065*z/(t0+273.15), 5.257)
    t = t0 - 0.0065*z
    rho = p/(2.87*(t0+273.15))
    return  {
        # hPa, 地上気圧
        'p0': p0,
        # ℃, 地上気温
        't0': t0,
        # hPa, 気圧
        'p': p,
        # ℃, 温度
        't': t,
        # kg/m3, 空気密度
        'rho': rho,
        # 空気の動粘性係数
        'nu': 1.4607e-5,
        }


def aircraft_mode(conf):
    """水平巡航時の推力推算
    """
    
    # m, 前後ファン距離
    l_fan = conf['l_ffan'] + conf['l_rfan']

    # 主翼
    # m2, 主翼翼面積
    S = conf['b'] * conf['c']
    # アスペクト比
    AR = conf['b']**2 / S
    if conf['is_biplane'] == True:
        # m2, 主翼濡れ面積
        S_wet = S * 4
        # m2, biplane の翼面積
        S = 2 * S
        # biplane のeffective アスペクト比
        AR = (1 + 8/np.pi*conf['h']/conf['b']) * AR
    else:
        # m2, 単葉の場合の濡れ面積
        S_wet = S * 2
    
    # 先尾翼
    # m2, 先尾翼の翼面積
    S_t = conf['b_t'] * conf['c_t']
    # 先尾翼のアスペクト比
    AR_t = conf['b_t']**2 / S_t
    # m2, 先尾翼の濡れ面積
    S_t_wet = S_t * 2

    # 摩擦係数
    # 胴体のレイノルズ数
    Re_f = conf['V_c']*conf['L']/air(conf['z'])['nu']
    # 胴体の摩擦係数
    Cf_f = 0.075 / (np.log10(Re_f)-2)**2
    # 主翼のレイノルズ数
    Re = conf['V_c']*conf['c']/air(conf['z'])['nu']
    # 主翼の摩擦係数
    Cf = 0.075 / (np.log10(Re)-2)**2

    # 胴体
    # m2, 胴体の正面面積
    S_0 = np.pi * (conf['b_f']/2)**2
    # m2, 胴体の濡れ面積 (楕円体だと近似)
    S_w = 3 * conf['L']/conf['b_f'] * S_0
    # 胴体の抵抗係数
    CDf_1 = 4.5e-3*S_w/S
    # 臨界Re以下, 1e5以下
    CDf_2 = (0.33*conf['b_f']/conf['L']+Cf_f*(3*conf['L']/conf['b_f']+3*np.sqrt(conf['b_f']/conf['L'])))*S_0/S
    # 全Re域
    CDf_3 = (3*conf['L']/conf['b_f']+4.5*np.sqrt(conf['b_f']/conf['L'])+21*(conf['b_f']/conf['L'])**2)*Cf_f*S_0/S
    CDf_4 = (0.44*conf['b_f']/conf['L']+4*Cf_f*(conf['L']/conf['b_f']+np.sqrt(conf['b_f']/conf['L'])))*S_0/S
    CDf_5 = 0.2*S_0/S
    CDf_6 = (1+np.sqrt(conf['b_f']/conf['L'])**3+0.11*(conf['b_f']/conf['L'])**2)*Cf_f*S_0/S
    #CDf_6 = (1+1.5*np.sqrt(conf['b_f']/conf['L'])**3+7*(conf['b_f']/conf['L'])**3)*Cf_f*S_0/S
    # 臨界Re以下
    CDf_7 = (Cf_f*(1+np.sqrt(conf['b_f']/conf['L'])**3)+0.15*(conf['b_f']/conf['L'])**2)*S_0/S
    # 胴体の抵抗係数概算, 文献??
    if conf['is_streamline'] == True:
        CDf = (CDf_1+CDf_3)/2 # 流線型と近似
    else:
        CDf = (CDf_4+CDf_5+CDf_7)/3 # 楕円形と近似

    # TODO ファンのケース この辺の扱い？
    # 前部ファンケース濡れ面積
    S_ffan_wet = (conf['b_ffan']*conf['c_ffan']+conf['c_ffan']*conf['h_ffan'])*4*2
    # 後部ファンケース濡れ面積
    S_rfan_wet = (conf['b_rfan']*conf['c_rfan']+conf['c_rfan']*conf['h_rfan'])*4*2

    # 揚力およびモーメント
    # N, 必要揚力
    L_total = conf['MTO']*9.8
    CL_total = L_total/(1/2*air(conf['z'])['rho']*conf['V_c']**2*S)
    # 先尾翼(揚力担当あり)
    # /rad, 2次元揚力傾斜
    a0_t = 2*np.pi
    # /rad, 3次元揚力傾斜
    a_t = a0_t/(1+a0_t/np.pi/AR_t/conf['e_t'])
    # NACA0012 を使用
    # ゼロ揚力係数
    alpha0_t = 0.0
    # 揚力係数
    CL_t = a_t*np.deg2rad((conf['i_t']-alpha0_t))
    # N, 先尾翼揚力 (なお，翼面積からは胴体と重なる部分を差し引いた)
    L_t = 1/2*air(conf['z'])['rho']*conf['V_c']**2*(S_t-conf['c_t']*conf['b_f'])*CL_t
    # モーメント係数
    Cm0_t = 0.0
    # Nm, 重心周りのモーメント
    Mcg_t = L_t*(conf['l_t']-conf['x_cg']) + 1/2*air(conf['z'])['rho']*conf['V_c']**2*(S_t-conf['c_t']*conf['b_f'])*conf['c_t']*Cm0_t
    # モーメントを主翼面積で無次元化
    Cmcg_t = (CL_t*(conf['l_t']-conf['x_cg'])/conf['c']+Cm0_t*conf['c_t']/conf['c'])*(S_t-conf['c_t']*conf['b_f'])/S
    # 主翼
    # /rad, 2次元揚力傾斜
    a0 = 2*np.pi
    # /rad, 3次元揚力傾斜
    a = a0/(1+a0/np.pi/AR/conf['e'])
    # NACA2412 を使用
    # ゼロ揚力係数
    alpha0 = -2
    # deg, 取り付け角, 先尾翼の揚力を考慮
    i = (CL_total-CL_t*S_t/S)/(a*np.pi/180) + alpha0
    # 揚力係数
    CL = a*np.deg2rad((i-alpha0))
    # N, 主翼揚力
    L_w = 1/2*air(conf['z'])['rho']*conf['V_c']**2*S*CL
    # モーメント係数
    Cm0 = -0.045
    # Nm, 重心周りのモーメント
    Mcg_w = -L_w*conf['x_cg'] + 1/2*air(conf['z'])['rho']*conf['V_c']**2*S*conf['c']*Cm0
    # 重心周りの無次元モーメント
    Cmcg_w = -CL*conf['x_cg']/conf['c']+Cm0

    # 全機
    # 重心周り全機モーメント
    Mcg = Mcg_t + Mcg_w
    Cmcg = Cmcg_t + Cmcg_w
    # 主翼の抗力係数
    CD0_w = S_wet*Cf/S
    # 先尾翼の抗力係数 (主翼面積で無次元化)
    CD0_t = S_t_wet*Cf/S
    # 主翼と先尾翼の抗力係数
    CD0_wt = CD0_w + CD0_t
    # 前後ファンケース抗力係数
    CD0_fan = (S_ffan_wet+S_rfan_wet)*Cf/S
    # 全機形状抗力係数
    CD0_total = CD0_wt + CDf + CD0_fan
    # 全機抗力係数 (主翼，先尾翼の誘導抵抗こみ)
    CD = CD0_total + CL**2/np.pi/AR/conf['e'] + CL_t**2/np.pi/AR_t/conf['e_t']*(S_t-conf['c_t']*conf['b_f'])/S
    # 揚抗比
    LD_ratio = CL_total/CD
    # N, 全機抗力
    D = 1/2*air(conf['z'])['rho']*conf['V_c']**2*S*CD
    # W, 必要パワー
    P = D*conf['V_c']
    return {
            'D': D,
            'P': P,
            }

def hover_fan(conf, T_need_all, V_c=0):
    """ホバリング時のファンの挙動
    Args:
        conf (dict): 条件 {loading_ratio: T_needのうち対象のファン群の担当比率, rho: 空気密度[kg/m3], n: 対象のファン群のファンの個数, b: ブレード枚数, R: ブレード半径[m], c: ブレード翼弦長[m], theta_0: ルートピッチ角[deg], theta_t: ブレード端ピッチ角[deg], a: ブレード揚力傾斜[/rad], Cd: 抗力係数, B: 翼端損失因子}
        T_need_all (float): N, 全機で望む推力
        V_c=0 (float): m/s, 巡航速度, 0の時ホバリング
    Returns:
        {Omega: 回転速度[rpm], T:推力[N], Q:トルク[Nm], P:パワー[W]}
    """
    # m^2, ローター円盤面積
    S = np.pi * conf['R']**2
    # ソリディティ, ヘリコプタ入門 4.24
    sigma = conf['b']*conf['c']/np.pi/conf['R']

    # ファン1個が出す必要がある推力
    T_need = T_need_all * conf['loading_ratio'] / conf['n']
    # m/s, 吹き下ろし速度 (ホバリング時), ヘリコプタ入門 4.1
    #v_0 = np.sqrt(T_need/2/conf['rho']/np.pi/conf['R']**2)
    # m/s, 吹き下ろし速度, 航空工学1 4.15
    v = -V_c/2 + np.sqrt((V_c/2)**2 + (T_need/(2*conf['rho']*S)))

    # 角速度
    Omega = np.arange(1000, 10000, 10)  # rpm
    omega = Omega / 60 * 2 * np.pi      # rad

    # m/s, 翼端の速度
    v_t = conf['R'] * omega
    # N/m^2 動圧
    q = conf['rho'] * S * v_t**2

    # 流入比, ヘリコプタ入門 5.8
    Lambda = v/omega/conf['R']
    # rad, 吹き下ろし角
    phi_t = -np.arctan(v/omega/conf['R'])

    # 推力係数, 翼素理論, 航空工学I 4.127a
    C_T_guess1 = conf['a']*sigma/2*(conf['B']**3 *np.deg2rad(conf['theta_0'])/3 - conf['B']**4 *np.deg2rad(conf['theta_t'])/4 - conf['B']**2 * Lambda/2) - sigma*conf['Cd']*Lambda/4
    # 推力係数, ヘリコプタ入門 4.47
    # 計算の参考？ ピッチ角theta_0で一定，吹き下ろし一定の仮定
    C_T_guess2 = conf['a']*sigma/4*(2/3*np.deg2rad(conf['theta_0'])+phi_t)
    # 推力係数, Newton法, ヘリコプタ入門 4.29
    #C_T = newton(lambda x: x-conf['a']*sigma/4*(conf['theta_t']-np.sqrt(x/2)), C_T_guess1)

    # 必要推力からomega を求める
    C_T_guess = C_T_guess1
    T_guess = q * C_T_guess

    err = abs(T_guess - T_need)
    err_min_idx = np.argmin(err)

    Omega = Omega[err_min_idx]
    omega = omega[err_min_idx]
    v_t = v_t[err_min_idx]
    q = q[err_min_idx]
    Lambda = Lambda[err_min_idx]
    phi_t = phi_t[err_min_idx]
    C_T_guess = C_T_guess[err_min_idx]
    T_guess = T_guess[err_min_idx]
    C_T = C_T_guess

    # C_T/sigma はローター効率と関係がある, ヘリコプタ入門4.7

    # N, 推力, ヘリコプタ入門 4.22
    T = q * C_T
    # トルク係数, パワ係数と同じ, ヘリコプタ入門 4.39
    C_Q = -phi_t*C_T + sigma*conf['Cd']/8
    # Nm, トルク, ヘリコプタ入門 4.36
    Q = q * conf['R'] * C_Q
    # W, パワー
    P = Q * omega
    return {'Omega': Omega,
            'T': T,
            'Q': Q,
            'P': P,
            'C_T': C_T}


def main():
    print(aircraft_mode(CONF))
    print(hover_fan(FRONT_FAN_CONF, CONF['MTO']*9.8))
    print(hover_fan(REAR_FAN_CONF, CONF['MTO']*9.8))
    #print(fan(FRONT_FAN_CONF, aircraft_mode(CONF)['D'], V_c=CONF['V_c']))
    #ファンの数こんなに少なくて巡航できるんだろうか謎
    REAR_FAN_CONF_CRUISE = copy(REAR_FAN_CONF)
    REAR_FAN_CONF_CRUISE['loading_ratio'] = 1
    REAR_FAN_CONF_CRUISE['n'] = 4
    print(hover_fan(REAR_FAN_CONF_CRUISE, aircraft_mode(CONF)['D'], V_c=CONF['V_c']))

    omega = np.arange(500, 10000, 10)
    cruise_fan_results = [momentum_blade_element_theory.cruise_fan(REAR_FAN_CONF_CRUISE, CONF['V_c'], x) for x in omega ]
    T_list = np.array([ res['T'] for res in cruise_fan_results] )
    err_min_idx = np.argmin(abs(T_list-aircraft_mode(CONF)['D']/REAR_FAN_CONF_CRUISE['n']))
    omega = omega[err_min_idx]
    eta = cruise_fan_results[err_min_idx]['eta']
    Q = cruise_fan_results[err_min_idx]['Q']
    print(omega, eta, Q*omega*REAR_FAN_CONF_CRUISE['n'])

## execution -----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
