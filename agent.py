from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def get_goal(self):
        pass

    @abstractmethod
    def set_goals(self, goals):
        pass

    @abstractmethod
    def get_u(self):
        pass

    @abstractmethod
    def step(self, u):
        pass