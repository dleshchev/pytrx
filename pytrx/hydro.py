import numpy as np



class HydroProperties:

    def __init__(self, density=None, molar_mass=None):
        self.density = density
        self.molar_mass = molar_mass

def add_solvent(d, keys, hp_obj):
    for key in keys:
        d[key] = hp_obj


acetonitrile = HydroProperties(density=786,                     # kg/m3
                               molar_mass=41.05e-3,             # kg/mol
                               )

data = {}

acetonitrile_keys = ['acetonitrile', 'Acetonitrile', 'acn', 'MeCN', 'ACN', 'CH3CN', 'ch3cn']
add_solvent(data, acetonitrile_keys, acetonitrile)


cyclohexane_keys = ['cyclohexane', 'c6h12', 'C6H12']
tetrahydrofuran_keys = ['thf', 'THF', 'Tetrahydrofuran', 'tetrahydrofuran']
water_keys = ['Water', 'water', 'h2o', 'H2O']