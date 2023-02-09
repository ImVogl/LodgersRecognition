import base_msg as message
import uuid
import numpy

# Message to send image
class ImageMessage(message.BaseMessage):

    # Constructor.
    def __init__(self, *arg) -> None:
        self.image = None
        super().__init__(str(uuid.UUID('ad205ef2-85ee-4f5b-9f3e-f8e2e7eaaba7')), *arg)

    # Setting target data
    def set_data(self, image: numpy.array):
        self.image = image
    
    # Initialization from json
    def set_data_from_json(self, json_data: dict):
        self.image = json_data["image"]
    
    # Convert this object to string message.
    def convert_to_json(self):
        if self.image is None:
            return { "id": self.id, "command": str(self.command), "image": self.image }

        return { "id": self.id, "command": str(self.command), "image": self.image.tolist() }
