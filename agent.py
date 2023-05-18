import pyautogui

class Agent():

    @property
    def position(self):
        return self.calculate_position()

    @property
    def weight(self):
        return self.calculate_weight()

    @staticmethod
    def capture(*args, **kwargs):
        return pyautogui.screenshot(*args, **kwargs)

    def calculate_position(self):
        image  = self.__class__.capture()
        

    def calculate_weight(self):
        image = self.__class__.capture(region=())
        
    @property
    def action(self):
        return self.get_current_action()
    
    def get_current_acction(self):
        return None
    
    @property
    def next_action(self):
        action_set = (
            self.position,
            self.weight,
            self.action
        )
        return self.send_to_cnn(action_set)
    
    def send_to_cnn(self):
        return None