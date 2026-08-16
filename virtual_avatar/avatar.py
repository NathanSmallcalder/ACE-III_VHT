from furhat_remote_api import FurhatRemoteAPI

def furhat_connect(host="127.0.0.1") -> FurhatRemoteAPI:
    """Build a client for the Furhat robot at `host`."""
    furhat = FurhatRemoteAPI(host)
    return furhat

def furhat_nod(furhat):
    furhat.gesture(name="Nod")

def furhat_repeat(furhat, angle=-15, duration=2):
    """ Warmer response for repeating questions titls head sideways while talking"""
    gesture = {
        "name": "HeadTiltConfused",
        "frames": [
            {"time": [duration],        "params": {"NECK_ROLL": angle}},  # reach tilt
            {"time": [duration + 1.5], "params": {"NECK_ROLL": angle}},  # hold
            {"time": [duration + 1.5 + duration], "params": {"NECK_ROLL": 0}}  # return to neutral
        ],
        "class": "furhatos.gestures.Gesture"
    }
    result = furhat.gesture(body=gesture, blocking=False)
    print(result)
    return result
