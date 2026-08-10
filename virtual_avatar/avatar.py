from furhat_remote_api import FurhatRemoteAPI

def furhat_connect(host="127.0.0.1") -> FurhatRemoteAPI:
    """Build a client for the Furhat robot at `host`."""
    furhat = FurhatRemoteAPI(host)
    print("[avatar] available voices:", furhat.get_voices())
    status = furhat.set_voice(name="en-US-JennyNeural")
    print("[avatar] set_voice status:", status)
    return furhat