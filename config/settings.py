class Config:
    # Camera Settings
    CAMERA_IDS = [0, 1, 2, 3]
    MAX_CAMERAS = 4
    FRAME_SIZE = (640, 480)
    
    # Detection Settings
    FACE_CONFIDENCE = 0.7
    ORDER_CONFIDENCE = 0.6
    ENTRANCE_LINE = 0.2
    EXIT_LINE = 0.8
    
    # UI Settings
    REFRESH_RATE = 100  # ms
    THEME = "dark"
    
config = Config()