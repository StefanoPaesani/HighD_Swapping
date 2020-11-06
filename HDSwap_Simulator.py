"""


Stefano Paesani, Nov. 2020
"""


from GaussianTransformations import *
import numpy as np



##########################################################################################
if __name__=="__main__":




    dim =
    s_par = 0.2
    num_modes = 2
    cov0 = np.identity(2*num_modes)
    cov = cov0


### Tests with Stefano's evolution method
    cov_new = cov0
    cov_new = symplectic_evolution(cov_new, get_tms_sym(s_par, 0, 0, 1, num_modes))


##### tests with strawberryfield

    eng = sf.Engine('gaussian')

    ##### test TMS
    prog2 = sf.Program(num_modes)
    with prog2.context as q:
        S2 = S2gate(s_par)
        S2 | (q[0], q[1])
    state2 = eng.run(prog2).state
    # print(state2.means())
    print(state2.cov())


    mapped_state =  map_into_StrawberryFields(np.zeros(num_modes), cov_new, num_modes)
    # print(mapped_state.means())
    print(mapped_state.cov())

    print(mapped_state == state2)




