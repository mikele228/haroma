# Lazy engine registry — import engines directly from their modules to avoid
# triggering heavy dependency chains (cv2, torch, etc.) at package level.
