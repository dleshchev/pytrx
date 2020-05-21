import numpy as np



class HydroProperties:

    def __init__(self, density=None, molar_mass=None):
        self.density = density
        self.molar_mass = molar_mass

def add_solvent(d, keys, hp_obj):
    for key in keys:
        d[key] = hp_obj


data = {}


acetonitrile = HydroProperties(density=786,                     # kg/m3
                               molar_mass=41.05e-3,             # kg/mol
                               )
acetonitrile_keys = ['acetonitrile', 'Acetonitrile', 'acn', 'MeCN', 'ACN', 'CH3CN', 'ch3cn']
add_solvent(data, acetonitrile_keys, acetonitrile)



cyclohexane = HydroProperties(density=779,                     # kg/m3
                               molar_mass=84.16e-3,             # kg/mol
                               )
cyclohexane_keys = ['cyclohexane', 'c6h12', 'C6H12']
add_solvent(data, cyclohexane, cyclohexane_keys)


tetrahydrofuran = HydroProperties(density=889,                     # kg/m3
                               molar_mass=72.11e-3,             # kg/mol
                               )
tetrahydrofuran_keys = ['thf', 'THF', 'Tetrahydrofuran', 'tetrahydrofuran']
add_solvent(data, tetrahydrofuran, tetrahydrofuran_keys)

water = HydroProperties(density=1000,                     # kg/m3
                               molar_mass=18.01528e-3,             # kg/mol
                               )
water_keys = ['Water', 'water', 'h2o', 'H2O']
add_solvent(data, water, water_keys)