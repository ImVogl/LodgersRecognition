from enum import Enum
from abc import ABC, abstractmethod
import uuid

# Type for messages.
class CommandType(Enum):
    SUBSCRIBE = 1
    GET = 2
    SET = 3

# Basic message
class BaseMessage(ABC):
    
    # Constructor.
    @abstractmethod
    def __init__(self, identifier: str, *arg) -> None:
        if len(arg) == 0:
            self.init_for_get_request()
        
        elif len(arg) == 1:
            if type(arg[0]) == 'dict':
                self.init_from_json(arg[0])
            else:
                self.init_for_set_request(arg[0])

        self.id = identifier
    
    # Initialisation for get request
    def init_for_get_request(self):
        self.command = CommandType.GET

    # Setting target data
    @abstractmethod
    def set_data(self, data):
        pass
    
    # Setting target data from JSON
    @abstractmethod
    def set_data_from_json(self, json: dict):
        pass
    
    # Initialization from arguments
    def init_for_set_request(self, data):
        self.command = CommandType.SET
        self.set_data(data)
    
    # Initialization from json
    def init_from_json(self, json_data: dict):
        self.command = json_data["command"]
        self.id = json_data["id"]
        self.set_data_from_json(json_data)

    # Changing command to subscribe
    def change_to_subscribe(self):
        self.command = CommandType.SUBSCRIBE
        
    # Convert this object to string message.
    @abstractmethod
    def convert_to_json(self, command):
        pass
