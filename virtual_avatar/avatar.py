from furhat_remote_api import FurhatRemoteAPI

def furhat_connect(host="127.0.0.1") -> FurhatRemoteAPI:
    """Build a client for the Furhat robot at `host`."""
    furhat = FurhatRemoteAPI(host)
    furhat.set_voice(name="Amy (en-GB) - Amazon Polly")
    return furhat