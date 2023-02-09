import base_msg as message
import uuid

# Message to send image
class RecognitionResult(message.BaseMessage):

    # Constructor.
    def __init__(self, *arg) -> None:
        self.object_id = -1
        super().__init__(str(uuid.UUID('8961ee74-2261-4634-a3f9-085c6d0fd910')), *arg)

    # Setting target data
    def set_data(self, object_id: int):
        self.object_id = object_id
    
    # Initialization from json
    def set_data_from_json(self, json_data: dict):
        self.object_id = json_data["object_id"]
    
    # Convert this object to string message.
    def convert_to_json(self):
        return { "id": self.id, "command": str(self.command), "object_id": self.object_id }

