import os

class Config:
    def __init__(self):
        self.QXToken = os.getenv('QXToken', 'Toke')
        self.SIMULATION = os.getenv('SIMULATION', 'True')