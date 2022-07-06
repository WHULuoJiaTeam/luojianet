import copy

# from ..distances import LpDistance
# from ..reducers import DoNothingReducer, MeanReducer, MultipleReducers
# from .module_with_records import ModuleWithRecords

from ..distances import LpDistance
from ..reducers import DoNothingReducer, MeanReducer, MultipleReducers
from ..utils.module_with_records import ModuleWithRecords


class ModuleWithRecordsAndReducer(ModuleWithRecords):
    def __init__(self, reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.set_reducer(reducer)

    def get_default_reducer(self):
        return MeanReducer()

    def set_reducer(self, reducer):
        if isinstance(reducer, (MultipleReducers, DoNothingReducer)):
            self.reducer = reducer
        elif len(self.sub_loss_names()) == 1:
            self.reducer = (
                self.get_default_reducer()
                if reducer is None
                else copy.deepcopy(reducer)
            )
        else:
            reducer_dict = {}
            for k in self.sub_loss_names():
                reducer_dict[k] = (
                    self.get_default_reducer()
                    if reducer is None
                    else copy.deepcopy(reducer)
                )
            self.reducer = MultipleReducers(reducer_dict)

    def sub_loss_names(self):
        return ["loss"]


class ModuleWithRecordsAndDistance(ModuleWithRecords):
    def __init__(self, distance=None, **kwargs):
        super().__init__(**kwargs)
        self.distance = self.get_default_distance() if distance is None else distance

    def get_default_distance(self):
        return LpDistance(p=2.0)


class ModuleWithRecordsReducerAndDistance(
    ModuleWithRecordsAndReducer, ModuleWithRecordsAndDistance
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
