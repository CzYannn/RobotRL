from .policy import Policy
from .rvoPolicy import RVOPolicy
from .nonCooperativePolicy import NonCooperativePolicy, NonCooperativePlanner
from .timePosPolicy import TimePositionPolicy
from .dwaPolicy import DWAPolicy
from .rvoPolicy import RVOPolicy





policy_dict = {
    # rule
    'noncoop': NonCooperativePolicy,
    'noncoopl': NonCooperativePlanner,
    'tpp': TimePositionPolicy,
    'DWA': DWAPolicy,
    'rvo': RVOPolicy,
}


