# implement everything related to NCMC

def perform_ncmc_simulations():
    
    # initiallize sampling from initial conformation
    
    # samples for 1 ns
    
    # (START): use sample after 1 ns to initialize a NEQ switching protocoll (with propagation and perturbation kernal) 
    # from a source level of theory to a target level of theory and collect work values
    
    # use work value in metropolis criteria to either acceptor or reject the new conformation at the target level of theory
    
    # if accepted: this new conformation is now used with the target level of theory to continue sampling (for 1ps), then (START) again
    
    # if rejected: neither the conformation nor the work value are used, the velocity reversed initial conformation is used to continue sampling (for 1ps), then (START) again  
    