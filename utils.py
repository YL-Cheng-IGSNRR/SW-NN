import numpy as np
def get_LST(BT, wvc, emis, coe):
    # according to the coefficient, get the LST
    BT_1 = BT[:, 0].reshape(-1, 1)
    BT_2 = BT[:, 1].reshape(-1, 1)
    emi_mean = np.mean(emis, axis=1).reshape(-1, 1)
    emi_mean_1 = (1 - emi_mean).reshape(-1,1) # (1 - emi_mean)
    emi_d = emis[:, 0].reshape(-1,1) - emis[:, 1].reshape(-1,1)

    # sobrino model
    d_BT = BT_1.reshape(-1,1) - BT_2.reshape(-1,1)
    d_BT_2 = d_BT.reshape(-1,1) * d_BT.reshape(-1,1)
    wvc_emis = wvc.reshape(-1,1) * emi_mean_1.reshape(-1,1)
    wvc_demi = wvc.reshape(-1,1) * emi_d.reshape(-1,1)

    LST_est = (BT_1 + coe[:,0].reshape(-1,1) * d_BT + coe[:,1].reshape(-1,1) * d_BT_2 + coe[:,2].reshape(-1,1) + 
                coe[:,3].reshape(-1,1) * emi_mean_1 + coe[:,4].reshape(-1,1) * wvc_emis + 
                coe[:,5].reshape(-1,1) * emi_d + coe[:,6].reshape(-1,1) * wvc_demi)
    return LST_est


    