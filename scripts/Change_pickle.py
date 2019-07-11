import pickle

class ChangePickle(object):
    def __init__(self):
        pass

    def change_lower_res(self, soil_obj, val_float):
        for i, obj in enumerate(soil_obj):
            soil_obj[i][1].lower_reservoir = val_float
        return soil_obj


    def set_initial_values_large_dataset(self, sm_val=1., str_val=20.5, soils_val=250.0):
        with open("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/sm.pickled") as file:
            sm = pickle.load(file)

        with open("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/soils.pickled") as file:
            soils = pickle.load(file)

        with open("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/streamflows.pickled") as file:
            str = pickle.load(file)

        sm[:] = sm_val  # Arbitrary
        str[:] = str_val  # This is very close to the correct initial value in september 2007

        with open("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/soils.pickled", "wb") as f:
            pickle.dump(self.change_lower_res(soils, soils_val), f)

        with open("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/sm.pickled", "wb") as f:
            pickle.dump(sm, f)

        with open("/home/zachary/workspace/dawuaphydroengine/docs/bitterroot_2016-2017/streamflows.pickled", "wb") as f:
            pickle.dump(str, f)

    def changepickle(self):
        self.set_initial_values_large_dataset(sm_val=1., str_val=20.5, soils_val=250.0)


changepickle = ChangePickle()
changepickle.changepickle()