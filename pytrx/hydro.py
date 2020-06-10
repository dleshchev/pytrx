import numpy as np



class HydroProperties:

    def __init__(self, density=None, molar_mass=None):
        self.density = density
        self.molar_mass = molar_mass
        #Z
        #xyz


def add_solvent(d, keys, hp_obj):
    for key in keys:
        d[key] = hp_obj


solvent_data = {}


acetonitrile = HydroProperties(density=786,                     # kg/m3
                               molar_mass=41.05e-3,             # kg/mol
                               )
acetonitrile.Z = np.array(['C', 'C', 'H', 'H', 'H', 'N'])
acetonitrile.xyz = np.array([[  0.000000,    0.000000,    0.006313],
                             [  0.000000,    0.000000,    1.462539],
                             [  1.024583,    0.000000,   -0.370908],
                             [ -0.512291,   -0.887315,   -0.370908],
                             [ -0.512291,    0.887315,   -0.370908],
                             [  0.000000,    0.000000,    2.615205]])
acetonitrile_keys = ['acetonitrile', 'Acetonitrile', 'acn', 'MeCN', 'ACN', 'CH3CN', 'ch3cn']
add_solvent(solvent_data, acetonitrile_keys, acetonitrile)


cyclohexane = HydroProperties(density=779,                     # kg/m3
                              molar_mass=84.16e-3,             # kg/mol
                              )
cyclohexane.Z = np.array(['C']*6 + ['H']*12)
cyclohexane.xyz = np.array([[-0.7613, -0.7573,  0.9857],
                            [ 0.7614, -0.7575,  0.9855],
                            [-1.2697,  0.5686,  0.4362],
                            [ 1.2693, -0.5690, -0.4379],
                            [-0.7627,  0.7570, -0.9872],
                            [ 0.7603,  0.7568, -0.9883],
                            [-1.1313, -0.8928,  2.0267],
                            [-1.1314, -1.5893,  0.3455],
                            [ 1.1315, -1.7280,  1.3860],
                            [ 1.1315,  0.0762,  1.6235],
                            [-2.3828,  0.5666,  0.4355],
                            [-0.8972,  1.4009,  1.0747],
                            [ 2.3828, -0.5672, -0.4356],
                            [ 0.8980, -1.4027, -1.0759],
                            [-1.1338,  1.7273, -1.3870],
                            [-1.1329, -0.0768, -1.6251],
                            [ 1.1290,  0.8924, -2.0303],
                            [ 1.1300,  1.5904, -0.3493]])
cyclohexane_keys = ['cyclohexane', 'c6h12', 'C6H12']
add_solvent(solvent_data, cyclohexane_keys, cyclohexane)


tetrahydrofuran = HydroProperties(density=889,                     # kg/m3
                                  molar_mass=72.11e-3,             # kg/mol
                                  )
tetrahydrofuran.Z = np.array(['O']*1 + ['C']*4 + ['H']*8)
# structure is taken from SI of Cryst. Growth Des., 2015, 15 (3), pp 1073–1081 DOI: 10.1021/cg501228w
tetrahydrofuran.xyz = np.array([[ 0.000760889, -0.000738525, -1.456851081],
                                [-0.652804737, -0.977671000, -0.666797608],
                                [-0.135276924, -0.754196802,  0.753993299],
                                [ 0.652965710,  0.977170235, -0.666602790],
                                [ 0.134622392,  0.755233684,  0.754352491],
                                [-1.737556125, -0.829010551, -0.720475115],
                                [-0.415216782, -1.967675331, -1.059040344],
                                [-0.850969390, -1.066076288,  1.514860077],
                                [ 0.796865925, -1.302852121,  0.910687964],
                                [ 1.737518733,  0.828788014, -0.720974308],
                                [ 0.414866234,  1.967002482, -1.058148213],
                                [ 0.850649795,  1.065510448,  1.514996321],
                                [-0.796426720,  1.304515754,  0.911888645]])
tetrahydrofuran_keys = ['thf', 'THF', 'Tetrahydrofuran', 'tetrahydrofuran']
add_solvent(solvent_data, tetrahydrofuran_keys, tetrahydrofuran)

water = HydroProperties(density=1000,                     # kg/m3
                        molar_mass=18.01528e-3,             # kg/mol
                        )
water.Z = np.array(['O'] * 1 + ['H'] * 2)
water.xyz = np.array([[ 0.000,  0.000,  0.000],
                      [ 0.928,  0.013,  0.246],
                      [-0.263, -0.899, -0.21 ]])
water_keys = ['Water', 'water', 'h2o', 'H2O']
add_solvent(solvent_data, water_keys, water)