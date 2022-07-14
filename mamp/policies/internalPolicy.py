from mamp.policies import Policy

class InternalPolicy(Policy):
    """ Convert an observation to an action completely within the environment (for model-based/pre-trained, simulated agents).

    Please see the possible subclasses at :ref:`all_internal_policies`.
    """
    def __init__(self, str="Internal"):
        Policy.__init__(self, str=str)
        self.type = "internal"

    def find_next_action(self, obs, agent):
        """ Use the provided inputs to select a commanded action [heading delta, speed]
            To be implemented by children.
        """
        raise NotImplementedError