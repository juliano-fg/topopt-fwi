#!/usr/bin/env python

class input_data:
    """Parameters to configure the FWI algorithm"""
    def __init__(self):
        # General Configuration ------------------------------------------------- #
              
        self.tofwi = True               # True = TO-based FWI
                                        # False = Traditional FWI
                                    
        self.example = 'b'              # Synthetic true model ('a','b','c' or 'd')
        
        self.true_salt_vel = 4.5        # Salt velocity for the true model (km/s)
        
        self.inverse_crime = True       # True = data from constant density model
                                        # False = data from variable density model
                                    
        self.r_s = 0.5                  # Filter radius for gradient smoothing (km)

        self.n_processes = 1            # Number of processes (int)
        
        # Initial Model --------------------------------------------------------- #   
        
        self.initial_slope = 0.8333     # Slope for the background velocity
                                        # (Exact slope is 0.8333)
        
        self.estimated_salt_vel = 4.5   # Salt velocity estimate (km/s)
        
        # TO-based FWI Configuration -------------------------------------------- #
        
        # (These variables have no influence on the inversion if the traditional 
        # FWI approach was chosen).
        
        self.update_background = False  # True = reconstruct background velocity
                                        # False = fixed background velocity

        self.r_a = 0.1                  # Filter radius for variable smoothing (km)

        self.q = 3                      # Penalty exponent for the interpolation 

        self.projection = True          # True = Heaviside projection active
                                        # False = Heaviside projection disabled
        # ----------------------------------------------------------------------- #
